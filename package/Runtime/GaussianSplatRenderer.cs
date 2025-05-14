// SPDX-License-Identifier: MIT

using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Profiling;
using Unity.Profiling.LowLevel;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Serialization;
using UnityEngine.XR;

namespace GaussianSplatting.Runtime
{
    internal class GaussianSplatRenderSystem
    {
        // ReSharper disable MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        internal static readonly ProfilerMarker kProfDraw     = new(ProfilerCategory.Render, "GaussianSplat.Draw", MarkerFlags.SampleGPU);
        internal static readonly ProfilerMarker kProfCompose  = new(ProfilerCategory.Render, "GaussianSplat.Compose", MarkerFlags.SampleGPU);
        internal static readonly ProfilerMarker kProfCalcView = new(ProfilerCategory.Render, "GaussianSplat.CalcView", MarkerFlags.SampleGPU);
        // ReSharper restore MemberCanBePrivate.Global

        public static GaussianSplatRenderSystem Instance => s_instance ??= new GaussianSplatRenderSystem();
        private static GaussianSplatRenderSystem s_instance;

        private readonly Dictionary<GaussianSplatRenderer, MaterialPropertyBlock> _splats = new();
        private readonly HashSet<Camera> _cameraCommandBuffersDone = new();
        private readonly List<(GaussianSplatRenderer, MaterialPropertyBlock)> _activeSplats = new();

        private CommandBuffer _commandBuffer;

        public void RegisterSplat(GaussianSplatRenderer r)
        {
            if (_splats.Count == 0)
            {
                if (GraphicsSettings.currentRenderPipeline == null)
                    Camera.onPreCull += OnPreCullCamera;
            }

            _splats.Add(r, new MaterialPropertyBlock());
        }

        public void UnregisterSplat(GaussianSplatRenderer r)
        {
            if (!_splats.ContainsKey(r))
                return;
            _splats.Remove(r);
            if (_splats.Count == 0)
            {
                if (_cameraCommandBuffersDone != null)
                {
                    if (_commandBuffer != null)
                    {
                        foreach (var cam in _cameraCommandBuffersDone)
                        {
                            if (cam)
                                cam.RemoveCommandBuffer(CameraEvent.BeforeForwardAlpha, _commandBuffer);
                        }
                    }
                    _cameraCommandBuffersDone.Clear();
                }

                _activeSplats.Clear();
                _commandBuffer?.Dispose();
                _commandBuffer = null;
                Camera.onPreCull -= OnPreCullCamera;
            }
        }

        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        public bool GatherSplatsForCamera(Camera cam)
        {
            if (cam.cameraType == CameraType.Preview)
                return false;
            // gather all active & valid splat objects
            _activeSplats.Clear();
            foreach (var kvp in _splats)
            {
                var gs = kvp.Key;
                if (gs == null || !gs.isActiveAndEnabled || !gs.HasValidSplat || !gs.HasValidRenderSetup)
                    continue;
                _activeSplats.Add((kvp.Key, kvp.Value));
            }
            if (_activeSplats.Count == 0)
                return false;

            // sort them by order and depth from camera
            var camTr = cam.transform;
            _activeSplats.Sort((a, b) =>
            {
                var orderA = a.Item1.RenderOrder;
                var orderB = b.Item1.RenderOrder;
                if (orderA != orderB)
                    return orderB.CompareTo(orderA);
                var trA = a.Item1.transform;
                var trB = b.Item1.transform;
                var posA = camTr.InverseTransformPoint(trA.position);
                var posB = camTr.InverseTransformPoint(trB.position);
                return posA.z.CompareTo(posB.z);
            });

            return true;
        }

        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        public Material SortAndRenderSplats(Camera cam, CommandBuffer cmb)
        {
            Material matComposite = null;
            foreach (var kvp in _activeSplats)
            {
                var gs = kvp.Item1;
                gs.EnsureMaterials();
                matComposite = gs.MatComposite;
                var mpb = kvp.Item2;

                // sort
                var matrix = gs.transform.localToWorldMatrix;
                if (gs.FrameCounter % gs.SortNthFrame == 0)
                    gs.SortPoints(cmb, cam, matrix);
                ++gs.FrameCounter;

                // cache view
                kvp.Item2.Clear();
                Material displayMat = gs.Mode switch
                {
                    GaussianSplatRenderer.RenderMode.DebugPoints => gs.MatDebugPoints,
                    GaussianSplatRenderer.RenderMode.DebugPointIndices => gs.MatDebugPoints,
                    GaussianSplatRenderer.RenderMode.DebugBoxes => gs.MatDebugBoxes,
                    GaussianSplatRenderer.RenderMode.DebugChunkBounds => gs.MatDebugBoxes,
                    _ => gs.MatSplats
                };
                if (displayMat == null)
                    continue;

                gs.SetSplatDataOnMaterial(mpb);
                mpb.SetBuffer(GaussianSplatRenderer.Props.kSplatChunks, gs.GPUChunks);

                mpb.SetBuffer(GaussianSplatRenderer.Props.kSplatViewData, gs.GPUView);

                mpb.SetBuffer(GaussianSplatRenderer.Props.kOrderBuffer, gs.GPUSortKeys);
                mpb.SetFloat(GaussianSplatRenderer.Props.kSplatScale, gs.SplatScale);
                mpb.SetFloat(GaussianSplatRenderer.Props.kSplatOpacityScale, gs.OpacityScale);
                mpb.SetFloat(GaussianSplatRenderer.Props.kSplatSize, gs.PointDisplaySize);
                mpb.SetInteger(GaussianSplatRenderer.Props.kSHOrder, gs.SHOrder);
                mpb.SetInteger(GaussianSplatRenderer.Props.kSHOnly, gs.SHOnly ? 1 : 0);
                mpb.SetInteger(GaussianSplatRenderer.Props.kDisplayIndex, gs.Mode == GaussianSplatRenderer.RenderMode.DebugPointIndices ? 1 : 0);
                mpb.SetInteger(GaussianSplatRenderer.Props.kDisplayChunks, gs.Mode == GaussianSplatRenderer.RenderMode.DebugChunkBounds ? 1 : 0);

                cmb.BeginSample(kProfCalcView);
                gs.CalcViewData(cmb, cam);
                cmb.EndSample(kProfCalcView);

                // draw
                int indexCount = 6;
                int instanceCount = gs.SplatCount;
                MeshTopology topology = MeshTopology.Triangles;
                if (gs.Mode is GaussianSplatRenderer.RenderMode.DebugBoxes or GaussianSplatRenderer.RenderMode.DebugChunkBounds)
                    indexCount = 36;
                if (gs.Mode == GaussianSplatRenderer.RenderMode.DebugChunkBounds)
                    instanceCount = gs.GPUChunksValid ? gs.GPUChunks.count : 0;

                cmb.BeginSample(kProfDraw);
                cmb.DrawProcedural(gs.GPUIndexBuffer, matrix, displayMat, 0, topology, indexCount, instanceCount, mpb);
                cmb.EndSample(kProfDraw);
            }
            return matComposite;
        }

        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        // ReSharper disable once UnusedMethodReturnValue.Global - used by HDRP/URP features that are not always compiled
        public CommandBuffer InitialClearCmdBuffer(Camera cam)
        {
            _commandBuffer ??= new CommandBuffer {name = "RenderGaussianSplats"};
            if (GraphicsSettings.currentRenderPipeline == null && cam != null && !_cameraCommandBuffersDone.Contains(cam))
            {
                cam.AddCommandBuffer(CameraEvent.BeforeForwardAlpha, _commandBuffer);
                _cameraCommandBuffersDone.Add(cam);
            }

            // get render target for all splats
            _commandBuffer.Clear();
            return _commandBuffer;
        }

        private void OnPreCullCamera(Camera cam)
        {
            if (!GatherSplatsForCamera(cam))
                return;

            InitialClearCmdBuffer(cam);

            _commandBuffer.GetTemporaryRT(GaussianSplatRenderer.Props.kGaussianSplatRT, -1, -1, 0, FilterMode.Point, GraphicsFormat.R16G16B16A16_SFloat);
            _commandBuffer.SetRenderTarget(GaussianSplatRenderer.Props.kGaussianSplatRT, BuiltinRenderTextureType.CurrentActive);
            _commandBuffer.ClearRenderTarget(RTClearFlags.Color, new Color(0, 0, 0, 0), 0, 0);

            // We only need this to determine whether we're rendering into backbuffer or not. However, detection this
            // way only works in BiRP so only do it here.
            _commandBuffer.SetGlobalTexture(GaussianSplatRenderer.Props.kCameraTargetTexture, BuiltinRenderTextureType.CameraTarget);

            // add sorting, view calc and drawing commands for each splat object
            Material matComposite = SortAndRenderSplats(cam, _commandBuffer);

            // compose
            _commandBuffer.BeginSample(kProfCompose);
            _commandBuffer.SetRenderTarget(BuiltinRenderTextureType.CameraTarget);
            _commandBuffer.DrawProcedural(Matrix4x4.identity, matComposite, 0, MeshTopology.Triangles, 3, 1);
            _commandBuffer.EndSample(kProfCompose);
            _commandBuffer.ReleaseTemporaryRT(GaussianSplatRenderer.Props.kGaussianSplatRT);
        }
    }

    [ExecuteInEditMode]
    public class GaussianSplatRenderer : MonoBehaviour
    {
        public enum RenderMode
        {
            Splats,
            DebugPoints,
            DebugPointIndices,
            DebugBoxes,
            DebugChunkBounds,
        }

        [Tooltip("Rendering order compared to other splats. Within same order splats are sorted by distance. Higher order splats render 'on top of' lower order splats.")]
        public int RenderOrder;
        [Range(0.1f, 2.0f)] [Tooltip("Additional scaling factor for the splats")]
        public float SplatScale = 1.0f;
        [Range(0.05f, 20.0f)]
        [Tooltip("Additional scaling factor for opacity")]
        public float OpacityScale = 1.0f;
        [Range(0, 3)] [Tooltip("Spherical Harmonics order to use")]
        public int SHOrder = 3;
        [Tooltip("Show only Spherical Harmonics contribution, using gray color")]
        public bool SHOnly;
        [Range(1,30)] [Tooltip("Sort splats only every N frames")]
        public int SortNthFrame = 1;

        public RenderMode Mode = RenderMode.Splats;
        [FormerlySerializedAs("_pointDisplaySize")]
        [Range(1.0f,15.0f)] 
        public float PointDisplaySize = 3.0f;

        public GaussianCutout[] Cutouts;

        public Shader ShaderSplats;
        public Shader ShaderComposite;
        public Shader ShaderDebugPoints;
        public Shader ShaderDebugBoxes;
        [Tooltip("Gaussian splatting compute shader")]
        public ComputeShader CsSplatUtilities;

        private int _splatCount; // initially same as splat count, but editing can change this
        private GraphicsBuffer _gpuSortDistances;
        internal GraphicsBuffer GPUSortKeys;
        private GraphicsBuffer _gpuPosData;
        private GraphicsBuffer _gpuOtherData;
        private GraphicsBuffer _gpuSHData;
        private Texture _gpuColorData;
        internal GraphicsBuffer GPUChunks;
        internal bool GPUChunksValid;
        internal GraphicsBuffer GPUView;
        internal GraphicsBuffer GPUIndexBuffer;

        // these buffers are only for splat editing, and are lazily created
        private GraphicsBuffer _gpuEditCutouts;
        private GraphicsBuffer _gpuEditCountsBounds;
        private GraphicsBuffer _gpuEditSelected;
        private GraphicsBuffer _gpuEditDeleted;
        private GraphicsBuffer _gpuEditSelectedMouseDown; // selection state at start of operation
        private GraphicsBuffer _gpuEditPosMouseDown;      // position state at start of operation
        private GraphicsBuffer _gpuEditOtherMouseDown;    // rotation/scale state at start of operation

        private GpuSorting _sorter;
        private GpuSorting.Args _sorterArgs;

        internal Material MatSplats;
        internal Material MatComposite;
        internal Material MatDebugPoints;
        internal Material MatDebugBoxes;

        internal int FrameCounter;
        private GaussianSplat _prevSplat;
        private Hash128 _prevHash;
        private bool _registered;

        private static readonly ProfilerMarker kSProfSort = new(ProfilerCategory.Render, "GaussianSplat.Sort", MarkerFlags.SampleGPU);

        internal static class Props
        {
            public static readonly int kSplatPos               = Shader.PropertyToID("_SplatPos");
            public static readonly int kSplatOther             = Shader.PropertyToID("_SplatOther");
            public static readonly int kSplatSH                = Shader.PropertyToID("_SplatSH");
            public static readonly int kSplatColor             = Shader.PropertyToID("_SplatColor");
            public static readonly int kSplatSelectedBits      = Shader.PropertyToID("_SplatSelectedBits");
            public static readonly int kSplatDeletedBits       = Shader.PropertyToID("_SplatDeletedBits");
            public static readonly int kSplatBitsValid         = Shader.PropertyToID("_SplatBitsValid");
            public static readonly int kSplatFormat            = Shader.PropertyToID("_SplatFormat");
            public static readonly int kSplatChunks            = Shader.PropertyToID("_SplatChunks");
            public static readonly int kSplatChunkCount        = Shader.PropertyToID("_SplatChunkCount");
            public static readonly int kSplatViewData          = Shader.PropertyToID("_SplatViewData");
            public static readonly int kOrderBuffer            = Shader.PropertyToID("_OrderBuffer");
            public static readonly int kSplatScale             = Shader.PropertyToID("_SplatScale");
            public static readonly int kSplatOpacityScale      = Shader.PropertyToID("_SplatOpacityScale");
            public static readonly int kSplatSize              = Shader.PropertyToID("_SplatSize");
            public static readonly int kSplatCount             = Shader.PropertyToID("_SplatCount");
            public static readonly int kSHOrder                = Shader.PropertyToID("_SHOrder");
            public static readonly int kSHOnly                 = Shader.PropertyToID("_SHOnly");
            public static readonly int kDisplayIndex           = Shader.PropertyToID("_DisplayIndex");
            public static readonly int kDisplayChunks          = Shader.PropertyToID("_DisplayChunks");
            public static readonly int kGaussianSplatRT        = Shader.PropertyToID("_GaussianSplatRT");
            public static readonly int kSplatSortKeys          = Shader.PropertyToID("_SplatSortKeys");
            public static readonly int kSplatSortDistances     = Shader.PropertyToID("_SplatSortDistances");
            public static readonly int kSrcBuffer              = Shader.PropertyToID("_SrcBuffer");
            public static readonly int kDstBuffer              = Shader.PropertyToID("_DstBuffer");
            public static readonly int kBufferSize             = Shader.PropertyToID("_BufferSize");
            public static readonly int kMatrixMv               = Shader.PropertyToID("_MatrixMV");
            public static readonly int kMatrixObjectToWorld    = Shader.PropertyToID("_MatrixObjectToWorld");
            public static readonly int kMatrixWorldToObject    = Shader.PropertyToID("_MatrixWorldToObject");
            public static readonly int kVecScreenParams        = Shader.PropertyToID("_VecScreenParams");
            public static readonly int kVecWorldSpaceCameraPos = Shader.PropertyToID("_VecWorldSpaceCameraPos");
            public static readonly int kCameraTargetTexture    = Shader.PropertyToID("_CameraTargetTexture");
            public static readonly int kSelectionCenter        = Shader.PropertyToID("_SelectionCenter");
            public static readonly int kSelectionDelta         = Shader.PropertyToID("_SelectionDelta");
            public static readonly int kSelectionDeltaRot      = Shader.PropertyToID("_SelectionDeltaRot");
            public static readonly int kSplatCutoutsCount      = Shader.PropertyToID("_SplatCutoutsCount");
            public static readonly int kSplatCutouts           = Shader.PropertyToID("_SplatCutouts");
            public static readonly int kSelectionMode          = Shader.PropertyToID("_SelectionMode");
            public static readonly int kSplatPosMouseDown      = Shader.PropertyToID("_SplatPosMouseDown");
            public static readonly int kSplatOtherMouseDown    = Shader.PropertyToID("_SplatOtherMouseDown");
        }

        [field: NonSerialized] public bool EditModified { get; private set; }
        [field: NonSerialized] public uint EditSelectedSplats { get; private set; }
        [field: NonSerialized] public uint EditDeletedSplats { get; private set; }
        [field: NonSerialized] public uint EditCutSplats { get; private set; }
        [field: NonSerialized] public Bounds EditSelectedBounds { get; private set; }

        public GaussianSplat Splat { get; set; }

        public int SplatCount => _splatCount;

        private enum KernelIndices
        {
            SetIndices,
            CalcDistances,
            CalcViewData,
            UpdateEditData,
            InitEditData,
            ClearBuffer,
            InvertSelection,
            SelectAll,
            OrBuffers,
            SelectionUpdate,
            TranslateSelection,
            RotateSelection,
            ScaleSelection,
            ExportData,
            CopySplats,
        }

        public bool HasValidSplat =>
            Splat != null &&
            Splat.SplatCount > 0 &&
            Splat.FormatVersion == GaussianSplat.kCurrentVersion &&
            Splat.PosData != null &&
            Splat.OtherData != null &&
            Splat.SHData != null &&
            Splat.ColorData != null;
        public bool HasValidRenderSetup => _gpuPosData != null && _gpuOtherData != null && GPUChunks != null;

        private const int kGpuViewDataSize = 40;

        private void CreateResourcesForSplat()
        {
            if (!HasValidSplat)
                return;

            _splatCount = Splat.SplatCount;
            _gpuPosData = new GraphicsBuffer(GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource, (int) (Splat.PosData.dataSize / 4), 4) { name = "GaussianPosData" };
            _gpuPosData.SetData(Splat.PosData.GetData<uint>());
            _gpuOtherData = new GraphicsBuffer(GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource, (int) (Splat.OtherData.dataSize / 4), 4) { name = "GaussianOtherData" };
            _gpuOtherData.SetData(Splat.OtherData.GetData<uint>());
            _gpuSHData = new GraphicsBuffer(GraphicsBuffer.Target.Raw, (int) (Splat.SHData.dataSize / 4), 4) { name = "GaussianSHData" };
            _gpuSHData.SetData(Splat.SHData.GetData<uint>());
            var (texWidth, texHeight) = GaussianSplat.CalcTextureSize(Splat.SplatCount);
            var texFormat = GaussianSplat.ColorFormatToGraphics(Splat.ColFormat);
            var tex = new Texture2D(texWidth, texHeight, texFormat, TextureCreationFlags.DontInitializePixels | TextureCreationFlags.IgnoreMipmapLimit | TextureCreationFlags.DontUploadUponCreate) { name = "GaussianColorData" };
            tex.SetPixelData(Splat.ColorData.GetData<byte>(), 0);
            tex.Apply(false, true);
            _gpuColorData = tex;
            if (Splat.ChunkData != null && Splat.ChunkData.dataSize != 0)
            {
                GPUChunks = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                    (int) (Splat.ChunkData.dataSize / UnsafeUtility.SizeOf<GaussianSplat.ChunkInfo>()),
                    UnsafeUtility.SizeOf<GaussianSplat.ChunkInfo>()) {name = "GaussianChunkData"};
                GPUChunks.SetData(Splat.ChunkData.GetData<GaussianSplat.ChunkInfo>());
                GPUChunksValid = true;
            }
            else
            {
                // just a dummy chunk buffer
                GPUChunks = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1,
                    UnsafeUtility.SizeOf<GaussianSplat.ChunkInfo>()) {name = "GaussianChunkData"};
                GPUChunksValid = false;
            }

            GPUView = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Splat.SplatCount, kGpuViewDataSize);
            GPUIndexBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Index, 36, 2);
            // cube indices, most often we use only the first quad
            GPUIndexBuffer.SetData(new ushort[]
            {
                0, 1, 2, 1, 3, 2,
                4, 6, 5, 5, 6, 7,
                0, 2, 4, 4, 2, 6,
                1, 5, 3, 5, 7, 3,
                0, 4, 1, 4, 5, 1,
                2, 3, 6, 3, 7, 6
            });

            InitSortBuffers(SplatCount);
        }

        private void InitSortBuffers(int count)
        {
            _gpuSortDistances?.Dispose();
            GPUSortKeys?.Dispose();
            _sorterArgs.Resources.Dispose();

            EnsureSorterAndRegister();

            _gpuSortDistances = new GraphicsBuffer(GraphicsBuffer.Target.Structured, count, 4) { name = "GaussianSplatSortDistances" };
            GPUSortKeys = new GraphicsBuffer(GraphicsBuffer.Target.Structured, count, 4) { name = "GaussianSplatSortIndices" };

            // init keys buffer to splat indices
            CsSplatUtilities.SetBuffer((int)KernelIndices.SetIndices, Props.kSplatSortKeys, GPUSortKeys);
            CsSplatUtilities.SetInt(Props.kSplatCount, _gpuSortDistances.count);
            CsSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.SetIndices, out uint gsX, out _, out _);
            CsSplatUtilities.Dispatch((int)KernelIndices.SetIndices, (_gpuSortDistances.count + (int)gsX - 1)/(int)gsX, 1, 1);

            _sorterArgs.InputKeys = _gpuSortDistances;
            _sorterArgs.InputValues = GPUSortKeys;
            _sorterArgs.Count = (uint)count;
            if (_sorter.Valid)
                _sorterArgs.Resources = GpuSorting.SupportResources.Load((uint)count);
        }

        private bool ResourcesAreSetUp => 
            ShaderSplats != null && 
            ShaderComposite != null && 
            ShaderDebugPoints != null &&
            ShaderDebugBoxes != null && 
            CsSplatUtilities != null && 
            SystemInfo.supportsComputeShaders;

        public void EnsureMaterials()
        {
            if (MatSplats == null && ResourcesAreSetUp)
            {
                MatSplats = new Material(ShaderSplats) {name = "GaussianSplats"};
                MatComposite = new Material(ShaderComposite) {name = "GaussianClearDstAlpha"};
                MatDebugPoints = new Material(ShaderDebugPoints) {name = "GaussianDebugPoints"};
                MatDebugBoxes = new Material(ShaderDebugBoxes) {name = "GaussianDebugBoxes"};
            }
        }

        public void EnsureSorterAndRegister()
        {
            if (_sorter == null && ResourcesAreSetUp)
            {
                _sorter = new GpuSorting(CsSplatUtilities);
            }

            if (!_registered && ResourcesAreSetUp)
            {
                GaussianSplatRenderSystem.Instance.RegisterSplat(this);
                _registered = true;
            }
        }

        public void OnEnable()
        {
            FrameCounter = 0;
            if (!ResourcesAreSetUp)
                return;

            EnsureMaterials();
            EnsureSorterAndRegister();

            CreateResourcesForSplat();
        }

        private void SetSplatDataOnCS(CommandBuffer cmb, KernelIndices kernel)
        {
            ComputeShader cs = CsSplatUtilities;
            int kernelIndex = (int) kernel;
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.kSplatPos, _gpuPosData);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.kSplatChunks, GPUChunks);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.kSplatOther, _gpuOtherData);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.kSplatSH, _gpuSHData);
            cmb.SetComputeTextureParam(cs, kernelIndex, Props.kSplatColor, _gpuColorData);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.kSplatSelectedBits, _gpuEditSelected ?? _gpuPosData);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.kSplatDeletedBits, _gpuEditDeleted ?? _gpuPosData);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.kSplatViewData, GPUView);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.kOrderBuffer, GPUSortKeys);

            cmb.SetComputeIntParam(cs, Props.kSplatBitsValid, _gpuEditSelected != null && _gpuEditDeleted != null ? 1 : 0);
            uint format = (uint)Splat.PosFormat | ((uint)Splat.ScaleFormat << 8) | ((uint)Splat.ShFormat << 16);
            cmb.SetComputeIntParam(cs, Props.kSplatFormat, (int)format);
            cmb.SetComputeIntParam(cs, Props.kSplatCount, _splatCount);
            cmb.SetComputeIntParam(cs, Props.kSplatChunkCount, GPUChunksValid ? GPUChunks.count : 0);

            UpdateCutoutsBuffer();
            cmb.SetComputeIntParam(cs, Props.kSplatCutoutsCount, Cutouts?.Length ?? 0);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.kSplatCutouts, _gpuEditCutouts);
        }

        internal void SetSplatDataOnMaterial(MaterialPropertyBlock mat)
        {
            mat.SetBuffer(Props.kSplatPos, _gpuPosData);
            mat.SetBuffer(Props.kSplatOther, _gpuOtherData);
            mat.SetBuffer(Props.kSplatSH, _gpuSHData);
            mat.SetTexture(Props.kSplatColor, _gpuColorData);
            mat.SetBuffer(Props.kSplatSelectedBits, _gpuEditSelected ?? _gpuPosData);
            mat.SetBuffer(Props.kSplatDeletedBits, _gpuEditDeleted ?? _gpuPosData);
            mat.SetInt(Props.kSplatBitsValid, _gpuEditSelected != null && _gpuEditDeleted != null ? 1 : 0);
            uint format = (uint)Splat.PosFormat | ((uint)Splat.ScaleFormat << 8) | ((uint)Splat.ShFormat << 16);
            mat.SetInteger(Props.kSplatFormat, (int)format);
            mat.SetInteger(Props.kSplatCount, _splatCount);
            mat.SetInteger(Props.kSplatChunkCount, GPUChunksValid ? GPUChunks.count : 0);
        }

        private static void DisposeBuffer(ref GraphicsBuffer buf)
        {
            buf?.Dispose();
            buf = null;
        }

        private void DisposeResourcesForSplat()
        {
            DestroyImmediate(_gpuColorData);

            DisposeBuffer(ref _gpuPosData);
            DisposeBuffer(ref _gpuOtherData);
            DisposeBuffer(ref _gpuSHData);
            DisposeBuffer(ref GPUChunks);

            DisposeBuffer(ref GPUView);
            DisposeBuffer(ref GPUIndexBuffer);
            DisposeBuffer(ref _gpuSortDistances);
            DisposeBuffer(ref GPUSortKeys);

            DisposeBuffer(ref _gpuEditSelectedMouseDown);
            DisposeBuffer(ref _gpuEditPosMouseDown);
            DisposeBuffer(ref _gpuEditOtherMouseDown);
            DisposeBuffer(ref _gpuEditSelected);
            DisposeBuffer(ref _gpuEditDeleted);
            DisposeBuffer(ref _gpuEditCountsBounds);
            DisposeBuffer(ref _gpuEditCutouts);

            _sorterArgs.Resources.Dispose();

            _splatCount = 0;
            GPUChunksValid = false;

            EditSelectedSplats = 0;
            EditDeletedSplats = 0;
            EditCutSplats = 0;
            EditModified = false;
            EditSelectedBounds = default;
        }

        public void OnDisable()
        {
            DisposeResourcesForSplat();
            GaussianSplatRenderSystem.Instance.UnregisterSplat(this);
            _registered = false;

            DestroyImmediate(MatSplats);
            DestroyImmediate(MatComposite);
            DestroyImmediate(MatDebugPoints);
            DestroyImmediate(MatDebugBoxes);
        }

        internal void CalcViewData(CommandBuffer cmb, Camera cam)
        {
            if (cam.cameraType == CameraType.Preview)
                return;

            var tr = transform;

            Matrix4x4 matView = cam.worldToCameraMatrix;
            Matrix4x4 matO2W = tr.localToWorldMatrix;
            Matrix4x4 matW2O = tr.worldToLocalMatrix;
            int screenW = cam.pixelWidth, screenH = cam.pixelHeight;
            int eyeW = XRSettings.eyeTextureWidth, eyeH = XRSettings.eyeTextureHeight;
            Vector4 screenPar = new Vector4(eyeW != 0 ? eyeW : screenW, eyeH != 0 ? eyeH : screenH, 0, 0);
            Vector4 camPos = cam.transform.position;

            // calculate view dependent data for each splat
            SetSplatDataOnCS(cmb, KernelIndices.CalcViewData);

            cmb.SetComputeMatrixParam(CsSplatUtilities, Props.kMatrixMv, matView * matO2W);
            cmb.SetComputeMatrixParam(CsSplatUtilities, Props.kMatrixObjectToWorld, matO2W);
            cmb.SetComputeMatrixParam(CsSplatUtilities, Props.kMatrixWorldToObject, matW2O);

            cmb.SetComputeVectorParam(CsSplatUtilities, Props.kVecScreenParams, screenPar);
            cmb.SetComputeVectorParam(CsSplatUtilities, Props.kVecWorldSpaceCameraPos, camPos);
            cmb.SetComputeFloatParam(CsSplatUtilities, Props.kSplatScale, SplatScale);
            cmb.SetComputeFloatParam(CsSplatUtilities, Props.kSplatOpacityScale, OpacityScale);
            cmb.SetComputeIntParam(CsSplatUtilities, Props.kSHOrder, SHOrder);
            cmb.SetComputeIntParam(CsSplatUtilities, Props.kSHOnly, SHOnly ? 1 : 0);

            CsSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.CalcViewData, out uint gsX, out _, out _);
            cmb.DispatchCompute(CsSplatUtilities, (int)KernelIndices.CalcViewData, (GPUView.count + (int)gsX - 1)/(int)gsX, 1, 1);
        }

        internal void SortPoints(CommandBuffer cmd, Camera cam, Matrix4x4 matrix)
        {
            if (cam.cameraType == CameraType.Preview)
                return;

            Matrix4x4 worldToCamMatrix = cam.worldToCameraMatrix;
            worldToCamMatrix.m20 *= -1;
            worldToCamMatrix.m21 *= -1;
            worldToCamMatrix.m22 *= -1;

            // calculate distance to the camera for each splat
            cmd.BeginSample(kSProfSort);
            cmd.SetComputeBufferParam(CsSplatUtilities, (int)KernelIndices.CalcDistances, Props.kSplatSortDistances, _gpuSortDistances);
            cmd.SetComputeBufferParam(CsSplatUtilities, (int)KernelIndices.CalcDistances, Props.kSplatSortKeys, GPUSortKeys);
            cmd.SetComputeBufferParam(CsSplatUtilities, (int)KernelIndices.CalcDistances, Props.kSplatChunks, GPUChunks);
            cmd.SetComputeBufferParam(CsSplatUtilities, (int)KernelIndices.CalcDistances, Props.kSplatPos, _gpuPosData);
            cmd.SetComputeIntParam(CsSplatUtilities, Props.kSplatFormat, (int)Splat.PosFormat);
            cmd.SetComputeMatrixParam(CsSplatUtilities, Props.kMatrixMv, worldToCamMatrix * matrix);
            cmd.SetComputeIntParam(CsSplatUtilities, Props.kSplatCount, _splatCount);
            cmd.SetComputeIntParam(CsSplatUtilities, Props.kSplatChunkCount, GPUChunksValid ? GPUChunks.count : 0);
            CsSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.CalcDistances, out uint gsX, out _, out _);
            cmd.DispatchCompute(CsSplatUtilities, (int)KernelIndices.CalcDistances, (_gpuSortDistances.count + (int)gsX - 1)/(int)gsX, 1, 1);

            // sort the splats
            EnsureSorterAndRegister();
            _sorter.Dispatch(cmd, _sorterArgs);
            cmd.EndSample(kSProfSort);
        }

        public void Update()
        {
            var curHash = Splat != null ? Splat.DataHash : new Hash128();
            if (_prevSplat != Splat || _prevHash != curHash)
            {
                _prevSplat = Splat;
                _prevHash = curHash;
                if (ResourcesAreSetUp)
                {
                    DisposeResourcesForSplat();
                    CreateResourcesForSplat();
                }
                else
                {
                    Debug.LogError($"{nameof(GaussianSplatRenderer)} component is not set up correctly (Resource references are missing), or platform does not support compute shaders");
                }
            }
        }

        public void ActivateCamera(int index)
        {
            Camera mainCam = Camera.main;
            if (!mainCam)
                return;
            if (Splat == null || Splat.Cameras == null)
                return;

            var selfTr = transform;
            var camTr = mainCam.transform;
            var prevParent = camTr.parent;
            var cam = Splat.Cameras[index];
            camTr.parent = selfTr;
            camTr.localPosition = cam.Pos;
            camTr.localRotation = Quaternion.LookRotation(cam.AxisZ, cam.AxisY);
            camTr.parent = prevParent;
            camTr.localScale = Vector3.one;
#if UNITY_EDITOR
            UnityEditor.EditorUtility.SetDirty(camTr);
#endif
        }

        private void ClearGraphicsBuffer(GraphicsBuffer buf)
        {
            CsSplatUtilities.SetBuffer((int)KernelIndices.ClearBuffer, Props.kDstBuffer, buf);
            CsSplatUtilities.SetInt(Props.kBufferSize, buf.count);
            CsSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.ClearBuffer, out uint gsX, out _, out _);
            CsSplatUtilities.Dispatch((int)KernelIndices.ClearBuffer, (int)((buf.count+gsX-1)/gsX), 1, 1);
        }

        private void UnionGraphicsBuffers(GraphicsBuffer dst, GraphicsBuffer src)
        {
            CsSplatUtilities.SetBuffer((int)KernelIndices.OrBuffers, Props.kSrcBuffer, src);
            CsSplatUtilities.SetBuffer((int)KernelIndices.OrBuffers, Props.kDstBuffer, dst);
            CsSplatUtilities.SetInt(Props.kBufferSize, dst.count);
            CsSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.OrBuffers, out uint gsX, out _, out _);
            CsSplatUtilities.Dispatch((int)KernelIndices.OrBuffers, (int)((dst.count+gsX-1)/gsX), 1, 1);
        }

        private static float SortableUintToFloat(uint v)
        {
            uint mask = ((v >> 31) - 1) | 0x80000000u;
            return math.asfloat(v ^ mask);
        }

        public void UpdateEditCountsAndBounds()
        {
            if (_gpuEditSelected == null)
            {
                EditSelectedSplats = 0;
                EditDeletedSplats = 0;
                EditCutSplats = 0;
                EditModified = false;
                EditSelectedBounds = default;
                return;
            }

            CsSplatUtilities.SetBuffer((int)KernelIndices.InitEditData, Props.kDstBuffer, _gpuEditCountsBounds);
            CsSplatUtilities.Dispatch((int)KernelIndices.InitEditData, 1, 1, 1);

            using CommandBuffer cmb = new CommandBuffer();
            SetSplatDataOnCS(cmb, KernelIndices.UpdateEditData);
            cmb.SetComputeBufferParam(CsSplatUtilities, (int)KernelIndices.UpdateEditData, Props.kDstBuffer, _gpuEditCountsBounds);
            cmb.SetComputeIntParam(CsSplatUtilities, Props.kBufferSize, _gpuEditSelected.count);
            CsSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.UpdateEditData, out uint gsX, out _, out _);
            cmb.DispatchCompute(CsSplatUtilities, (int)KernelIndices.UpdateEditData, (int)((_gpuEditSelected.count+gsX-1)/gsX), 1, 1);
            Graphics.ExecuteCommandBuffer(cmb);

            uint[] res = new uint[_gpuEditCountsBounds.count];
            _gpuEditCountsBounds.GetData(res);
            EditSelectedSplats = res[0];
            EditDeletedSplats = res[1];
            EditCutSplats = res[2];
            Vector3 min = new Vector3(SortableUintToFloat(res[3]), SortableUintToFloat(res[4]), SortableUintToFloat(res[5]));
            Vector3 max = new Vector3(SortableUintToFloat(res[6]), SortableUintToFloat(res[7]), SortableUintToFloat(res[8]));
            Bounds bounds = default;
            bounds.SetMinMax(min, max);
            if (bounds.extents.sqrMagnitude < 0.01)
                bounds.extents = new Vector3(0.1f,0.1f,0.1f);
            EditSelectedBounds = bounds;
        }

        private void UpdateCutoutsBuffer()
        {
            int bufferSize = Cutouts?.Length ?? 0;
            if (bufferSize == 0)
                bufferSize = 1;
            if (_gpuEditCutouts == null || _gpuEditCutouts.count != bufferSize)
            {
                _gpuEditCutouts?.Dispose();
                _gpuEditCutouts = new GraphicsBuffer(GraphicsBuffer.Target.Structured, bufferSize, UnsafeUtility.SizeOf<GaussianCutout.ShaderData>()) { name = "GaussianCutouts" };
            }

            NativeArray<GaussianCutout.ShaderData> data = new(bufferSize, Allocator.Temp);
            if (Cutouts != null)
            {
                var matrix = transform.localToWorldMatrix;
                for (var i = 0; i < Cutouts.Length; ++i)
                {
                    data[i] = GaussianCutout.GetShaderData(Cutouts[i], matrix);
                }
            }

            _gpuEditCutouts.SetData(data);
            data.Dispose();
        }

        private bool EnsureEditingBuffers()
        {
            if (!HasValidSplat || !HasValidRenderSetup)
                return false;

            if (_gpuEditSelected == null)
            {
                var target = GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource |
                             GraphicsBuffer.Target.CopyDestination;
                var size = (_splatCount + 31) / 32;
                _gpuEditSelected = new GraphicsBuffer(target, size, 4) {name = "GaussianSplatSelected"};
                _gpuEditSelectedMouseDown = new GraphicsBuffer(target, size, 4) {name = "GaussianSplatSelectedInit"};
                _gpuEditDeleted = new GraphicsBuffer(target, size, 4) {name = "GaussianSplatDeleted"};
                _gpuEditCountsBounds = new GraphicsBuffer(target, 3 + 6, 4) {name = "GaussianSplatEditData"}; // selected count, deleted bound, cut count, float3 min, float3 max
                ClearGraphicsBuffer(_gpuEditSelected);
                ClearGraphicsBuffer(_gpuEditSelectedMouseDown);
                ClearGraphicsBuffer(_gpuEditDeleted);
            }
            return _gpuEditSelected != null;
        }

        public void EditStoreSelectionMouseDown()
        {
            if (!EnsureEditingBuffers()) return;
            Graphics.CopyBuffer(_gpuEditSelected, _gpuEditSelectedMouseDown);
        }

        public void EditStorePosMouseDown()
        {
            if (_gpuEditPosMouseDown == null)
            {
                _gpuEditPosMouseDown = new GraphicsBuffer(_gpuPosData.target | GraphicsBuffer.Target.CopyDestination, _gpuPosData.count, _gpuPosData.stride) {name = "GaussianSplatEditPosMouseDown"};
            }
            Graphics.CopyBuffer(_gpuPosData, _gpuEditPosMouseDown);
        }
        public void EditStoreOtherMouseDown()
        {
            if (_gpuEditOtherMouseDown == null)
            {
                _gpuEditOtherMouseDown = new GraphicsBuffer(_gpuOtherData.target | GraphicsBuffer.Target.CopyDestination, _gpuOtherData.count, _gpuOtherData.stride) {name = "GaussianSplatEditOtherMouseDown"};
            }
            Graphics.CopyBuffer(_gpuOtherData, _gpuEditOtherMouseDown);
        }

        public void EditUpdateSelection(Vector2 rectMin, Vector2 rectMax, Camera cam, bool subtract)
        {
            if (!EnsureEditingBuffers()) return;

            Graphics.CopyBuffer(_gpuEditSelectedMouseDown, _gpuEditSelected);

            var tr = transform;
            Matrix4x4 matView = cam.worldToCameraMatrix;
            Matrix4x4 matO2W = tr.localToWorldMatrix;
            Matrix4x4 matW2O = tr.worldToLocalMatrix;
            int screenW = cam.pixelWidth, screenH = cam.pixelHeight;
            Vector4 screenPar = new Vector4(screenW, screenH, 0, 0);
            Vector4 camPos = cam.transform.position;

            using var cmb = new CommandBuffer();
            cmb.name = "SplatSelectionUpdate";
            SetSplatDataOnCS(cmb, KernelIndices.SelectionUpdate);

            cmb.SetComputeMatrixParam(CsSplatUtilities, Props.kMatrixMv, matView * matO2W);
            cmb.SetComputeMatrixParam(CsSplatUtilities, Props.kMatrixObjectToWorld, matO2W);
            cmb.SetComputeMatrixParam(CsSplatUtilities, Props.kMatrixWorldToObject, matW2O);

            cmb.SetComputeVectorParam(CsSplatUtilities, Props.kVecScreenParams, screenPar);
            cmb.SetComputeVectorParam(CsSplatUtilities, Props.kVecWorldSpaceCameraPos, camPos);

            cmb.SetComputeVectorParam(CsSplatUtilities, "_SelectionRect", new Vector4(rectMin.x, rectMax.y, rectMax.x, rectMin.y));
            cmb.SetComputeIntParam(CsSplatUtilities, Props.kSelectionMode, subtract ? 0 : 1);

            DispatchUtilsAndExecute(cmb, KernelIndices.SelectionUpdate, _splatCount);
            UpdateEditCountsAndBounds();
        }

        public void EditTranslateSelection(Vector3 localSpacePosDelta)
        {
            if (!EnsureEditingBuffers()) return;

            using var cmb = new CommandBuffer();
            cmb.name = "SplatTranslateSelection";
            SetSplatDataOnCS(cmb, KernelIndices.TranslateSelection);

            cmb.SetComputeVectorParam(CsSplatUtilities, Props.kSelectionDelta, localSpacePosDelta);

            DispatchUtilsAndExecute(cmb, KernelIndices.TranslateSelection, _splatCount);
            UpdateEditCountsAndBounds();
            EditModified = true;
        }

        public void EditRotateSelection(Vector3 localSpaceCenter, Matrix4x4 localToWorld, Matrix4x4 worldToLocal, Quaternion rotation)
        {
            if (!EnsureEditingBuffers()) return;
            if (_gpuEditPosMouseDown == null || _gpuEditOtherMouseDown == null) return; // should have captured initial state

            using var cmb = new CommandBuffer();
            cmb.name = "SplatRotateSelection";
            SetSplatDataOnCS(cmb, KernelIndices.RotateSelection);

            cmb.SetComputeBufferParam(CsSplatUtilities, (int)KernelIndices.RotateSelection, Props.kSplatPosMouseDown, _gpuEditPosMouseDown);
            cmb.SetComputeBufferParam(CsSplatUtilities, (int)KernelIndices.RotateSelection, Props.kSplatOtherMouseDown, _gpuEditOtherMouseDown);
            cmb.SetComputeVectorParam(CsSplatUtilities, Props.kSelectionCenter, localSpaceCenter);
            cmb.SetComputeMatrixParam(CsSplatUtilities, Props.kMatrixObjectToWorld, localToWorld);
            cmb.SetComputeMatrixParam(CsSplatUtilities, Props.kMatrixWorldToObject, worldToLocal);
            cmb.SetComputeVectorParam(CsSplatUtilities, Props.kSelectionDeltaRot, new Vector4(rotation.x, rotation.y, rotation.z, rotation.w));

            DispatchUtilsAndExecute(cmb, KernelIndices.RotateSelection, _splatCount);
            UpdateEditCountsAndBounds();
            EditModified = true;
        }


        public void EditScaleSelection(Vector3 localSpaceCenter, Matrix4x4 localToWorld, Matrix4x4 worldToLocal, Vector3 scale)
        {
            if (!EnsureEditingBuffers()) return;
            if (_gpuEditPosMouseDown == null) return; // should have captured initial state

            using var cmb = new CommandBuffer();
            cmb.name = "SplatScaleSelection";
            SetSplatDataOnCS(cmb, KernelIndices.ScaleSelection);

            cmb.SetComputeBufferParam(CsSplatUtilities, (int)KernelIndices.ScaleSelection, Props.kSplatPosMouseDown, _gpuEditPosMouseDown);
            cmb.SetComputeVectorParam(CsSplatUtilities, Props.kSelectionCenter, localSpaceCenter);
            cmb.SetComputeMatrixParam(CsSplatUtilities, Props.kMatrixObjectToWorld, localToWorld);
            cmb.SetComputeMatrixParam(CsSplatUtilities, Props.kMatrixWorldToObject, worldToLocal);
            cmb.SetComputeVectorParam(CsSplatUtilities, Props.kSelectionDelta, scale);

            DispatchUtilsAndExecute(cmb, KernelIndices.ScaleSelection, _splatCount);
            UpdateEditCountsAndBounds();
            EditModified = true;
        }

        public void EditDeleteSelected()
        {
            if (!EnsureEditingBuffers()) return;
            UnionGraphicsBuffers(_gpuEditDeleted, _gpuEditSelected);
            EditDeselectAll();
            UpdateEditCountsAndBounds();
            if (EditDeletedSplats != 0)
                EditModified = true;
        }

        public void EditSelectAll()
        {
            if (!EnsureEditingBuffers()) return;
            using var cmb = new CommandBuffer();
            cmb.name = "SplatSelectAll";
            SetSplatDataOnCS(cmb, KernelIndices.SelectAll);
            cmb.SetComputeBufferParam(CsSplatUtilities, (int)KernelIndices.SelectAll, Props.kDstBuffer, _gpuEditSelected);
            cmb.SetComputeIntParam(CsSplatUtilities, Props.kBufferSize, _gpuEditSelected.count);
            DispatchUtilsAndExecute(cmb, KernelIndices.SelectAll, _gpuEditSelected.count);
            UpdateEditCountsAndBounds();
        }

        public void EditDeselectAll()
        {
            if (!EnsureEditingBuffers()) return;
            ClearGraphicsBuffer(_gpuEditSelected);
            UpdateEditCountsAndBounds();
        }

        public void EditInvertSelection()
        {
            if (!EnsureEditingBuffers()) return;

            using var cmb = new CommandBuffer();
            cmb.name = "SplatInvertSelection";
            SetSplatDataOnCS(cmb, KernelIndices.InvertSelection);
            cmb.SetComputeBufferParam(CsSplatUtilities, (int)KernelIndices.InvertSelection, Props.kDstBuffer, _gpuEditSelected);
            cmb.SetComputeIntParam(CsSplatUtilities, Props.kBufferSize, _gpuEditSelected.count);
            DispatchUtilsAndExecute(cmb, KernelIndices.InvertSelection, _gpuEditSelected.count);
            UpdateEditCountsAndBounds();
        }

        public bool EditExportData(GraphicsBuffer dstData, bool bakeTransform)
        {
            if (!EnsureEditingBuffers()) return false;

            int flags = 0;
            var tr = transform;
            Quaternion bakeRot = tr.localRotation;
            Vector3 bakeScale = tr.localScale;

            if (bakeTransform)
                flags = 1;

            using var cmb = new CommandBuffer();
            cmb.name = "SplatExportData";
            SetSplatDataOnCS(cmb, KernelIndices.ExportData);
            cmb.SetComputeIntParam(CsSplatUtilities, "_ExportTransformFlags", flags);
            cmb.SetComputeVectorParam(CsSplatUtilities, "_ExportTransformRotation", new Vector4(bakeRot.x, bakeRot.y, bakeRot.z, bakeRot.w));
            cmb.SetComputeVectorParam(CsSplatUtilities, "_ExportTransformScale", bakeScale);
            cmb.SetComputeMatrixParam(CsSplatUtilities, Props.kMatrixObjectToWorld, tr.localToWorldMatrix);
            cmb.SetComputeBufferParam(CsSplatUtilities, (int)KernelIndices.ExportData, "_ExportBuffer", dstData);

            DispatchUtilsAndExecute(cmb, KernelIndices.ExportData, _splatCount);
            return true;
        }

        public void EditSetSplatCount(int newSplatCount)
        {
            if (newSplatCount <= 0 || newSplatCount > GaussianSplat.kMaxSplats)
            {
                Debug.LogError($"Invalid new splat count: {newSplatCount}");
                return;
            }
            if (Splat.ChunkData != null)
            {
                Debug.LogError("Only splats with VeryHigh quality can be resized");
                return;
            }
            if (newSplatCount == SplatCount)
                return;

            int posStride = (int)(Splat.PosData.dataSize / Splat.SplatCount);
            int otherStride = (int)(Splat.OtherData.dataSize / Splat.SplatCount);
            int shStride = (int) (Splat.SHData.dataSize / Splat.SplatCount);

            // create new GPU buffers
            var newPosData = new GraphicsBuffer(GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource, newSplatCount * posStride / 4, 4) { name = "GaussianPosData" };
            var newOtherData = new GraphicsBuffer(GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource, newSplatCount * otherStride / 4, 4) { name = "GaussianOtherData" };
            var newSHData = new GraphicsBuffer(GraphicsBuffer.Target.Raw, newSplatCount * shStride / 4, 4) { name = "GaussianSHData" };

            // new texture is a RenderTexture so we can write to it from a compute shader
            var (texWidth, texHeight) = GaussianSplat.CalcTextureSize(newSplatCount);
            var texFormat = GaussianSplat.ColorFormatToGraphics(Splat.ColFormat);
            var newColorData = new RenderTexture(texWidth, texHeight, texFormat, GraphicsFormat.None) { name = "GaussianColorData", enableRandomWrite = true };
            newColorData.Create();

            // selected/deleted buffers
            var selTarget = GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource | GraphicsBuffer.Target.CopyDestination;
            var selSize = (newSplatCount + 31) / 32;
            var newEditSelected = new GraphicsBuffer(selTarget, selSize, 4) {name = "GaussianSplatSelected"};
            var newEditSelectedMouseDown = new GraphicsBuffer(selTarget, selSize, 4) {name = "GaussianSplatSelectedInit"};
            var newEditDeleted = new GraphicsBuffer(selTarget, selSize, 4) {name = "GaussianSplatDeleted"};
            ClearGraphicsBuffer(newEditSelected);
            ClearGraphicsBuffer(newEditSelectedMouseDown);
            ClearGraphicsBuffer(newEditDeleted);

            var newGpuView = new GraphicsBuffer(GraphicsBuffer.Target.Structured, newSplatCount, kGpuViewDataSize);
            InitSortBuffers(newSplatCount);

            // copy existing data over into new buffers
            EditCopySplats(transform, newPosData, newOtherData, newSHData, newColorData, newEditDeleted, newSplatCount, 0, 0, _splatCount);

            // use the new buffers and the new splat count
            _gpuPosData.Dispose();
            _gpuOtherData.Dispose();
            _gpuSHData.Dispose();
            DestroyImmediate(_gpuColorData);
            GPUView.Dispose();

            _gpuEditSelected?.Dispose();
            _gpuEditSelectedMouseDown?.Dispose();
            _gpuEditDeleted?.Dispose();

            _gpuPosData = newPosData;
            _gpuOtherData = newOtherData;
            _gpuSHData = newSHData;
            _gpuColorData = newColorData;
            GPUView = newGpuView;
            _gpuEditSelected = newEditSelected;
            _gpuEditSelectedMouseDown = newEditSelectedMouseDown;
            _gpuEditDeleted = newEditDeleted;

            DisposeBuffer(ref _gpuEditPosMouseDown);
            DisposeBuffer(ref _gpuEditOtherMouseDown);

            _splatCount = newSplatCount;
            EditModified = true;
        }

        public void EditCopySplatsInto(GaussianSplatRenderer dst, int copySrcStartIndex, int copyDstStartIndex, int copyCount)
        {
            EditCopySplats(
                dst.transform,
                dst._gpuPosData, dst._gpuOtherData, dst._gpuSHData, dst._gpuColorData, dst._gpuEditDeleted,
                dst.SplatCount,
                copySrcStartIndex, copyDstStartIndex, copyCount);
            dst.EditModified = true;
        }

        public void EditCopySplats(
            Transform dstTransform,
            GraphicsBuffer dstPos, GraphicsBuffer dstOther, GraphicsBuffer dstSH, Texture dstColor,
            GraphicsBuffer dstEditDeleted,
            int dstSize,
            int copySrcStartIndex, int copyDstStartIndex, int copyCount)
        {
            if (!EnsureEditingBuffers()) return;

            Matrix4x4 copyMatrix = dstTransform.worldToLocalMatrix * transform.localToWorldMatrix;
            Quaternion copyRot = copyMatrix.rotation;
            Vector3 copyScale = copyMatrix.lossyScale;

            using var cmb = new CommandBuffer();
            cmb.name = "SplatCopy";
            SetSplatDataOnCS(cmb, KernelIndices.CopySplats);

            cmb.SetComputeBufferParam(CsSplatUtilities, (int)KernelIndices.CopySplats, "_CopyDstPos", dstPos);
            cmb.SetComputeBufferParam(CsSplatUtilities, (int)KernelIndices.CopySplats, "_CopyDstOther", dstOther);
            cmb.SetComputeBufferParam(CsSplatUtilities, (int)KernelIndices.CopySplats, "_CopyDstSH", dstSH);
            cmb.SetComputeTextureParam(CsSplatUtilities, (int)KernelIndices.CopySplats, "_CopyDstColor", dstColor);
            cmb.SetComputeBufferParam(CsSplatUtilities, (int)KernelIndices.CopySplats, "_CopyDstEditDeleted", dstEditDeleted);

            cmb.SetComputeIntParam(CsSplatUtilities, "_CopyDstSize", dstSize);
            cmb.SetComputeIntParam(CsSplatUtilities, "_CopySrcStartIndex", copySrcStartIndex);
            cmb.SetComputeIntParam(CsSplatUtilities, "_CopyDstStartIndex", copyDstStartIndex);
            cmb.SetComputeIntParam(CsSplatUtilities, "_CopyCount", copyCount);

            cmb.SetComputeVectorParam(CsSplatUtilities, "_CopyTransformRotation", new Vector4(copyRot.x, copyRot.y, copyRot.z, copyRot.w));
            cmb.SetComputeVectorParam(CsSplatUtilities, "_CopyTransformScale", copyScale);
            cmb.SetComputeMatrixParam(CsSplatUtilities, "_CopyTransformMatrix", copyMatrix);

            DispatchUtilsAndExecute(cmb, KernelIndices.CopySplats, copyCount);
        }

        private void DispatchUtilsAndExecute(CommandBuffer cmb, KernelIndices kernel, int count)
        {
            CsSplatUtilities.GetKernelThreadGroupSizes((int)kernel, out uint gsX, out _, out _);
            cmb.DispatchCompute(CsSplatUtilities, (int)kernel, (int)((count + gsX - 1)/gsX), 1, 1);
            Graphics.ExecuteCommandBuffer(cmb);
        }

        public GraphicsBuffer GpuEditDeleted => _gpuEditDeleted;
    }
}