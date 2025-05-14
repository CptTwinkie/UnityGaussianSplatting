using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Rendering;
// ReSharper disable PrivateFieldCanBeConvertedToLocalVariable

namespace GaussianSplatting.Runtime
{
    // GPU (uint key, uint payload) 8 bit-LSD radix sort, using reduce-then-scan
    // Copyright Thomas Smith 2024, MIT license
    // https://github.com/b0nes164/GPUSorting

    public class GpuSorting
    {
        //The size of a threadblock partition in the sort
        private const uint DEVICE_RADIX_SORT_PARTITION_SIZE = 3840;

        //The size of our radix in bits
        private const uint DEVICE_RADIX_SORT_BITS = 8;

        //Number of digits in our radix, 1 << DEVICE_RADIX_SORT_BITS
        private const uint DEVICE_RADIX_SORT_RADIX = 256;

        //Number of sorting passes required to sort a 32bit key, KEY_BITS / DEVICE_RADIX_SORT_BITS
        private const uint DEVICE_RADIX_SORT_PASSES = 4;

        //Keywords to enable for the shader
#pragma warning disable IDE0044 // Add readonly modifier
        private LocalKeyword _keyUintKeyword;
        private LocalKeyword _payloadUintKeyword;
        private LocalKeyword _ascendKeyword;
        private LocalKeyword _sortPairKeyword;
        private LocalKeyword _vulkanKeyword;
#pragma warning restore IDE0044 // Add readonly modifier

        public struct Args
        {
            public uint Count;
            public GraphicsBuffer InputKeys;
            public GraphicsBuffer InputValues;
            public SupportResources Resources;
            internal int WorkGroupCount;
        }

        public struct SupportResources
        {
            public GraphicsBuffer AltBuffer;
            public GraphicsBuffer AltPayloadBuffer;
            public GraphicsBuffer PassHistBuffer;
            public GraphicsBuffer GlobalHistBuffer;

            public static SupportResources Load(uint count)
            {
                //This is threadBlocks * DEVICE_RADIX_SORT_RADIX
                uint scratchBufferSize = DivRoundUp(count, DEVICE_RADIX_SORT_PARTITION_SIZE) * DEVICE_RADIX_SORT_RADIX; 
                uint reducedScratchBufferSize = DEVICE_RADIX_SORT_RADIX * DEVICE_RADIX_SORT_PASSES;

                var target = GraphicsBuffer.Target.Structured;
                var resources = new SupportResources
                {
                    AltBuffer = new GraphicsBuffer(target, (int)count, 4) { name = "DeviceRadixAlt" },
                    AltPayloadBuffer = new GraphicsBuffer(target, (int)count, 4) { name = "DeviceRadixAltPayload" },
                    PassHistBuffer = new GraphicsBuffer(target, (int)scratchBufferSize, 4) { name = "DeviceRadixPassHistogram" },
                    GlobalHistBuffer = new GraphicsBuffer(target, (int)reducedScratchBufferSize, 4) { name = "DeviceRadixGlobalHistogram" },
                };
                return resources;
            }

            public void Dispose()
            {
                AltBuffer?.Dispose();
                AltPayloadBuffer?.Dispose();
                PassHistBuffer?.Dispose();
                GlobalHistBuffer?.Dispose();

                AltBuffer = null;
                AltPayloadBuffer = null;
                PassHistBuffer = null;
                GlobalHistBuffer = null;
            }
        }

        private readonly ComputeShader _cs;
        private readonly int _kernelInitDeviceRadixSort = -1;
        private readonly int _kernelUpsweep = -1;
        private readonly int _kernelScan = -1;
        private readonly int _kernelDownsweep = -1;

        private readonly bool _valid;

        public bool Valid => _valid;

        public GpuSorting(ComputeShader cs)
        {
            _cs = cs;
            if (cs)
            {
                _kernelInitDeviceRadixSort = cs.FindKernel("InitDeviceRadixSort");
                _kernelUpsweep = cs.FindKernel("Upsweep");
                _kernelScan = cs.FindKernel("Scan");
                _kernelDownsweep = cs.FindKernel("Downsweep");
            }

            _valid = _kernelInitDeviceRadixSort >= 0 &&
                      _kernelUpsweep >= 0 &&
                      _kernelScan >= 0 &&
                      _kernelDownsweep >= 0;
            if (_valid)
            {
                if (!cs.IsSupported(_kernelInitDeviceRadixSort) ||
                    !cs.IsSupported(_kernelUpsweep) ||
                    !cs.IsSupported(_kernelScan) ||
                    !cs.IsSupported(_kernelDownsweep))
                {
                    _valid = false;
                }
            }

            _keyUintKeyword = new LocalKeyword(cs, "KEY_UINT");
            _payloadUintKeyword = new LocalKeyword(cs, "PAYLOAD_UINT");
            _ascendKeyword = new LocalKeyword(cs, "SHOULD_ASCEND");
            _sortPairKeyword = new LocalKeyword(cs, "SORT_PAIRS");
            _vulkanKeyword = new LocalKeyword(cs, "VULKAN");

            cs.EnableKeyword(_keyUintKeyword);
            cs.EnableKeyword(_payloadUintKeyword);
            cs.EnableKeyword(_ascendKeyword);
            cs.EnableKeyword(_sortPairKeyword);
            if (SystemInfo.graphicsDeviceType == GraphicsDeviceType.Vulkan)
            {
                cs.EnableKeyword(_vulkanKeyword);
            }
            else
            {
                cs.DisableKeyword(_vulkanKeyword);
            }
        }

        private static uint DivRoundUp(uint x, uint y) => (x + y - 1) / y;

        //Can we remove the last 4 padding without breaking?
        private struct SortConstants
        {
            public uint NumKeys;      // The number of keys to sort
            public uint RadixShift;   // The radix shift value for the current pass
            public uint ThreadBlocks; // threadBlocks
            public uint Padding0;     // Padding - unused
        }

        public void Dispatch(CommandBuffer cmd, Args args)
        {
            Assert.IsTrue(Valid);

            GraphicsBuffer srcKeyBuffer = args.InputKeys;
            GraphicsBuffer srcPayloadBuffer = args.InputValues;
            GraphicsBuffer dstKeyBuffer = args.Resources.AltBuffer;
            GraphicsBuffer dstPayloadBuffer = args.Resources.AltPayloadBuffer;

            SortConstants constants = default;
            constants.NumKeys = args.Count;
            constants.ThreadBlocks = DivRoundUp(args.Count, DEVICE_RADIX_SORT_PARTITION_SIZE);

            // Setup overall constants
            cmd.SetComputeIntParam(_cs, "e_numKeys", (int)constants.NumKeys);
            cmd.SetComputeIntParam(_cs, "e_threadBlocks", (int)constants.ThreadBlocks);

            //Set statically located buffers
            //Upsweep
            cmd.SetComputeBufferParam(_cs, _kernelUpsweep, "b_passHist", args.Resources.PassHistBuffer);
            cmd.SetComputeBufferParam(_cs, _kernelUpsweep, "b_globalHist", args.Resources.GlobalHistBuffer);

            //Scan
            cmd.SetComputeBufferParam(_cs, _kernelScan, "b_passHist", args.Resources.PassHistBuffer);

            //Downsweep
            cmd.SetComputeBufferParam(_cs, _kernelDownsweep, "b_passHist", args.Resources.PassHistBuffer);
            cmd.SetComputeBufferParam(_cs, _kernelDownsweep, "b_globalHist", args.Resources.GlobalHistBuffer);

            //Clear the global histogram
            cmd.SetComputeBufferParam(_cs, _kernelInitDeviceRadixSort, "b_globalHist", args.Resources.GlobalHistBuffer);
            cmd.DispatchCompute(_cs, _kernelInitDeviceRadixSort, 1, 1, 1);

            // Execute the sort algorithm in 8-bit increments
            for (constants.RadixShift = 0; constants.RadixShift < 32; constants.RadixShift += DEVICE_RADIX_SORT_BITS)
            {
                cmd.SetComputeIntParam(_cs, "e_radixShift", (int)constants.RadixShift);

                //Upsweep
                cmd.SetComputeBufferParam(_cs, _kernelUpsweep, "b_sort", srcKeyBuffer);
                cmd.DispatchCompute(_cs, _kernelUpsweep, (int)constants.ThreadBlocks, 1, 1);

                // Scan
                cmd.DispatchCompute(_cs, _kernelScan, (int)DEVICE_RADIX_SORT_RADIX, 1, 1);

                // Downsweep
                cmd.SetComputeBufferParam(_cs, _kernelDownsweep, "b_sort", srcKeyBuffer);
                cmd.SetComputeBufferParam(_cs, _kernelDownsweep, "b_sortPayload", srcPayloadBuffer);
                cmd.SetComputeBufferParam(_cs, _kernelDownsweep, "b_alt", dstKeyBuffer);
                cmd.SetComputeBufferParam(_cs, _kernelDownsweep, "b_altPayload", dstPayloadBuffer);
                cmd.DispatchCompute(_cs, _kernelDownsweep, (int)constants.ThreadBlocks, 1, 1);

                // Swap
                (srcKeyBuffer, dstKeyBuffer) = (dstKeyBuffer, srcKeyBuffer);
                (srcPayloadBuffer, dstPayloadBuffer) = (dstPayloadBuffer, srcPayloadBuffer);
            }
        }
    }
}
