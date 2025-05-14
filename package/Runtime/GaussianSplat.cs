// SPDX-License-Identifier: MIT

using System;
using System.IO;
using Newtonsoft.Json;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Serialization;

namespace GaussianSplatting.Runtime
{
    [JsonObject(MemberSerialization.OptIn)]
    public class GaussianSplat
    {
        #region Data

        public const int kCurrentVersion = 2023_10_20;
        public const int kChunkSize = 256;
        public const int kTextureWidth = 2048; // allows up to 32M splats on desktop GPU (2k width x 16k height)
        public const int kMaxSplats = 8_600_000; // mostly due to 2GB GPU buffer size limit when exporting a splat (2GB / 248B is just over 8.6M)

        // Serialized fields retained for UI information only
        [JsonProperty] public VectorFormat PosFormat = VectorFormat.Norm11;
        [JsonProperty] public VectorFormat ScaleFormat = VectorFormat.Norm11;
        [JsonProperty] public SHFormat ShFormat = SHFormat.Norm11;
        [JsonProperty] public ColorFormat ColFormat;
        [SerializeField] private TextAsset _posData;
        [SerializeField] private TextAsset _colorData;
        [SerializeField] private TextAsset _otherData;
        [SerializeField] private TextAsset _shData;
        /// <summary> Chunk data is optional (if data formats are fully lossless then there's no chunking) </summary>
        [SerializeField] private TextAsset _chunkData;
        [JsonProperty] public CameraInfo[] Cameras;
        [JsonProperty] public int FormatVersion;
        [JsonProperty] public int SplatCount;
        [JsonConverter(typeof(JsonVector3Converter))]
        [JsonProperty] public Vector3 BoundsMin;
        [JsonConverter(typeof(JsonVector3Converter))]
        [JsonProperty] public Vector3 BoundsMax;
        [JsonConverter(typeof(JsonHash128Converter))]
        [JsonProperty] public Hash128 DataHash;

        /// <summary> Retained from ScriptableObject for convenience. </summary>
        public string Name;

        [JsonProperty] public string PosDataPath;
        [JsonProperty] public string ColorDataPath;
        [JsonProperty] public string OtherDataPath;
        [JsonProperty] public string SHDataPath;
        [JsonProperty] public string ChunkDataPath;

        /*
         * Serialize
          FormatVersion (int)
          DataHash (string)
          SplatCount (int)
          BoundsMin (Vector3)
          BoundsMax (Vector3)
          PosFormat (int)
          ScaleFormat (int)
          SHFormat (int)
          ColFormat (int)
          PosDataPath (string)
          ColorDataPath (string)
          OtherDataPath (string)
          SHDataPath (string)
          ChunkDataPath (string)
          Cameras (array of CameraInfo)
        */

        public bool IsUsingChunks =>
            PosFormat   != VectorFormat.Float32 ||
            ScaleFormat != VectorFormat.Float32 ||
            ColFormat   != ColorFormat.Float32X4 ||
            ShFormat    != SHFormat.Float32;

        public TextAsset PosData => _posData;
        public TextAsset ColorData => _colorData;
        public TextAsset OtherData => _otherData;
        public TextAsset SHData => _shData;
        public TextAsset ChunkData => _chunkData;

        #endregion Data

        #region Types

        /// <summary> Match VECTOR_FMT_* in HLSL </summary>
        public enum VectorFormat
        {
            Float32, // 12 bytes: 32F.32F.32F
            Norm16, // 6 bytes: 16.16.16
            Norm11, // 4 bytes: 11.10.11
            Norm6   // 2 bytes: 6.5.5
        }

        public static int GetVectorSize(VectorFormat fmt)
        {
            return fmt switch
            {
                VectorFormat.Float32 => 12,
                VectorFormat.Norm16 => 6,
                VectorFormat.Norm11 => 4,
                VectorFormat.Norm6 => 2,
                _ => throw new ArgumentOutOfRangeException(nameof(fmt), fmt, null)
            };
        }

        public enum ColorFormat
        {
            Float32X4,
            Float16X4,
            Norm8X4,
            BC7,
        }

        public static int GetColorSize(ColorFormat fmt)
        {
            return fmt switch
            {
                ColorFormat.Float32X4 => 16,
                ColorFormat.Float16X4 => 8,
                ColorFormat.Norm8X4 => 4,
                ColorFormat.BC7 => 1,
                _ => throw new ArgumentOutOfRangeException(nameof(fmt), fmt, null)
            };
        }

        public enum SHFormat
        {
            Float32,
            Float16,
            Norm11,
            Norm6,
            Cluster64K,
            Cluster32K,
            Cluster16K,
            Cluster8K,
            Cluster4K,
        }

        public struct SHTableItemFloat32
        {
            public float3 SH1, SH2, SH3, SH4, SH5, SH6, SH7, SH8, SH9, SHA, SHB, SHC, SHD, SHE, SHF;
            public float3 SHPadding; // pad to multiple of 16 bytes
        }
        public struct SHTableItemFloat16
        {
            public half3 SH1, SH2, SH3, SH4, SH5, SH6, SH7, SH8, SH9, SHA, SHB, SHC, SHD, SHE, SHF;
            public half3 SHPadding; // pad to multiple of 16 bytes
        }
        public struct SHTableItemNorm11
        {
            public uint SH1, SH2, SH3, SH4, SH5, SH6, SH7, SH8, SH9, SHA, SHB, SHC, SHD, SHE, SHF;
        }
        public struct SHTableItemNorm6
        {
            public ushort SH1, SH2, SH3, SH4, SH5, SH6, SH7, SH8, SH9, SHA, SHB, SHC, SHD, SHE, SHF;
            public ushort SHPadding; // pad to multiple of 4 bytes
        }

        public struct ChunkInfo
        {
            public uint ColR, ColG, ColB, ColA;
            public float2 PosX, PosY, PosZ;
            public uint SclX, SclY, SclZ;
            public uint SHR, SHG, SHB;
        }

        [Serializable]
        public struct CameraInfo
        {
            [FormerlySerializedAs("pos")] public Vector3 Pos;
            [FormerlySerializedAs("axisX")] public Vector3 AxisX;
            [FormerlySerializedAs("axisY")] public Vector3 AxisY;
            [FormerlySerializedAs("axisZ")] public Vector3 AxisZ;
            [FormerlySerializedAs("fov")] public float FOV;
        }

        #endregion Types

        public void Initialize(int splats, VectorFormat formatPos, VectorFormat formatScale, ColorFormat formatColor, SHFormat formatSh, Vector3 bMin, Vector3 bMax, CameraInfo[] cameraInfos)
        {
            SplatCount = splats;
            FormatVersion = kCurrentVersion;
            PosFormat = formatPos;
            ScaleFormat = formatScale;
            ColFormat = formatColor;
            ShFormat = formatSh;
            Cameras = cameraInfos;
            BoundsMin = bMin;
            BoundsMax = bMax;
        }

        public void SetSplatFiles()
        {
            _chunkData = IsUsingChunks ? new TextAsset(File.ReadAllBytes(ChunkDataPath)) : null;
            _posData = new TextAsset(File.ReadAllBytes(PosDataPath));
            _otherData = new TextAsset(File.ReadAllBytes(OtherDataPath));
            _colorData = new TextAsset(File.ReadAllBytes(ColorDataPath));
            _shData = new TextAsset(File.ReadAllBytes(SHDataPath));
        }

        public void SetDataHash(Hash128 hash)
        {
            DataHash = hash;
        }

        public static int GetOtherSizeNoSHIndex(VectorFormat scaleFormat)
        {
            return 4 + GetVectorSize(scaleFormat);
        }

        public static int GetSHCount(SHFormat fmt, int splatCount)
        {
            return fmt switch
            {
                SHFormat.Float32 => splatCount,
                SHFormat.Float16 => splatCount,
                SHFormat.Norm11 => splatCount,
                SHFormat.Norm6 => splatCount,
                SHFormat.Cluster64K => 64 * 1024,
                SHFormat.Cluster32K => 32 * 1024,
                SHFormat.Cluster16K => 16 * 1024,
                SHFormat.Cluster8K => 8 * 1024,
                SHFormat.Cluster4K => 4 * 1024,
                _ => throw new ArgumentOutOfRangeException(nameof(fmt), fmt, null)
            };
        }

        public static (int,int) CalcTextureSize(int splatCount)
        {
            int width = kTextureWidth;
            int height = math.max(1, (splatCount + width - 1) / width);
            // our swizzle tiles are 16x16, so make texture multiple of that height
            int blockHeight = 16;
            height = (height + blockHeight - 1) / blockHeight * blockHeight;
            return (width, height);
        }

        public static GraphicsFormat ColorFormatToGraphics(ColorFormat format)
        {
            return format switch
            {
                ColorFormat.Float32X4 => GraphicsFormat.R32G32B32A32_SFloat,
                ColorFormat.Float16X4 => GraphicsFormat.R16G16B16A16_SFloat,
                ColorFormat.Norm8X4 => GraphicsFormat.R8G8B8A8_UNorm,
                ColorFormat.BC7 => GraphicsFormat.RGBA_BC7_UNorm,
                _ => throw new ArgumentOutOfRangeException(nameof(format), format, null)
            };
        }

        public static long CalcPosDataSize(int splatCount, VectorFormat formatPos)
        {
            return splatCount * GetVectorSize(formatPos);
        }
        public static long CalcOtherDataSize(int splatCount, VectorFormat formatScale)
        {
            return splatCount * GetOtherSizeNoSHIndex(formatScale);
        }
        public static long CalcColorDataSize(int splatCount, ColorFormat formatColor)
        {
            var (width, height) = CalcTextureSize(splatCount);
            return width * height * GetColorSize(formatColor);
        }
        public static long CalcSHDataSize(int splatCount, SHFormat formatSh)
        {
            int shCount = GetSHCount(formatSh, splatCount);
            return formatSh switch
            {
                SHFormat.Float32 => shCount * UnsafeUtility.SizeOf<SHTableItemFloat32>(),
                SHFormat.Float16 => shCount * UnsafeUtility.SizeOf<SHTableItemFloat16>(),
                SHFormat.Norm11 => shCount * UnsafeUtility.SizeOf<SHTableItemNorm11>(),
                SHFormat.Norm6 => shCount * UnsafeUtility.SizeOf<SHTableItemNorm6>(),
                _ => shCount * UnsafeUtility.SizeOf<SHTableItemFloat16>() + splatCount * 2
            };
        }
        public static long CalcChunkDataSize(int splatCount)
        {
            int chunkCount = (splatCount + kChunkSize - 1) / kChunkSize;
            return chunkCount * UnsafeUtility.SizeOf<ChunkInfo>();
        }
    }
}
