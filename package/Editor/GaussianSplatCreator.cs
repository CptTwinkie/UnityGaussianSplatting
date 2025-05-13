// SPDX-License-Identifier: MIT

using System;
using System.Collections.Generic;
using System.IO;
using GaussianSplatting.Editor.Utils;
using GaussianSplatting.Runtime;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;
using UnityEngine.Experimental.Rendering;

namespace GaussianSplatting.Editor
{
    [BurstCompile]
    public class GaussianSplatCreator : EditorWindow
    {
        private const string kProgressTitle = "Creating Gaussian Splat";
        private const string kCamerasJson = "cameras.json";
        private const string kPrefQuality = "nesnausk.GaussianSplatting.CreatorQuality";
        private const string kPrefOutputFolder = "nesnausk.GaussianSplatting.CreatorOutputFolder";

        private enum DataQuality
        {
            VeryHigh,
            High,
            Medium,
            Low,
            VeryLow,
            Custom,
        }

        private readonly FilePickerControl _filePicker = new();

        [SerializeField] private string _inputFile;
        [SerializeField] private bool _importCameras = true;
        [SerializeField] private string _outputFolder = "/";
        [SerializeField] private DataQuality _quality = DataQuality.Medium;
        [SerializeField] private GaussianSplat.VectorFormat _formatPos;
        [SerializeField] private GaussianSplat.VectorFormat _formatScale;
        [SerializeField] private GaussianSplat.ColorFormat _formatColor;
        [SerializeField] private GaussianSplat.SHFormat _formatSH;

        private string _errorMessage;
        private string _prevFilePath;
        private int _prevVertexCount;
        private long _prevFileSize;

        private bool IsUsingChunks =>
            _formatPos   != GaussianSplat.VectorFormat.Float32 ||
            _formatScale != GaussianSplat.VectorFormat.Float32 ||
            _formatColor != GaussianSplat.ColorFormat.Float32X4 ||
            _formatSH    != GaussianSplat.SHFormat.Float32;

        [MenuItem("Tools/Gaussian Splats/Create GaussianSplat")]
        public static void Init()
        {
            var window = GetWindowWithRect<GaussianSplatCreator>(new Rect(50, 50, 360, 340), false, "Gaussian Splat Creator", true);
            window.minSize = new Vector2(320, 320);
            window.maxSize = new Vector2(1500, 1500);
            window.Show();
        }

        private void Awake()
        {
            _quality = (DataQuality)EditorPrefs.GetInt(kPrefQuality, (int)DataQuality.Medium);
            _outputFolder = EditorPrefs.GetString(kPrefOutputFolder, "/");
        }

        private void OnEnable()
        {
            ApplyQualityLevel();
        }

        private void OnGUI()
        {
            EditorGUILayout.Space();
            GUILayout.Label("Input data", EditorStyles.boldLabel);
            var rect = EditorGUILayout.GetControlRect(true);
            _inputFile = _filePicker.PathFieldGUI(rect, new GUIContent("Input PLY/SPZ File"), _inputFile, "ply,spz", "PointCloudFile");
            _importCameras = EditorGUILayout.Toggle("Import Cameras", _importCameras);

            if (_inputFile != _prevFilePath && !string.IsNullOrWhiteSpace(_inputFile))
            {
                _prevVertexCount = 0;
                _errorMessage = null;
                try
                {
                    _prevVertexCount = GaussianFileReader.ReadFileHeader(_inputFile);
                }
                catch (Exception ex)
                {
                    _errorMessage = ex.Message;
                }

                _prevFileSize = File.Exists(_inputFile) ? new FileInfo(_inputFile).Length : 0;
                _prevFilePath = _inputFile;
            }

            if (_prevVertexCount > 0)
                EditorGUILayout.LabelField("File Size", $"{EditorUtility.FormatBytes(_prevFileSize)} - {_prevVertexCount:N0} splats");
            else
                GUILayout.Space(EditorGUIUtility.singleLineHeight);

            EditorGUILayout.Space();
            GUILayout.Label("Output", EditorStyles.boldLabel);
            rect = EditorGUILayout.GetControlRect(true);
            string newOutputFolder = _filePicker.PathFieldGUI(rect, new GUIContent("Output Folder"), _outputFolder, null, "GaussianOutputFolder");
            if (newOutputFolder != _outputFolder)
            {
                _outputFolder = newOutputFolder;
                EditorPrefs.SetString(kPrefOutputFolder, _outputFolder);
            }

            var newQuality = (DataQuality) EditorGUILayout.EnumPopup("Quality", _quality);
            if (newQuality != _quality)
            {
                _quality = newQuality;
                EditorPrefs.SetInt(kPrefQuality, (int)_quality);
                ApplyQualityLevel();
            }

            long sizePos = 0, sizeOther = 0, sizeCol = 0, sizeSHs = 0, totalSize = 0;
            if (_prevVertexCount > 0)
            {
                sizePos = GaussianSplat.CalcPosDataSize(_prevVertexCount, _formatPos);
                sizeOther = GaussianSplat.CalcOtherDataSize(_prevVertexCount, _formatScale);
                sizeCol = GaussianSplat.CalcColorDataSize(_prevVertexCount, _formatColor);
                sizeSHs = GaussianSplat.CalcSHDataSize(_prevVertexCount, _formatSH);
                long sizeChunk = IsUsingChunks ? GaussianSplat.CalcChunkDataSize(_prevVertexCount) : 0;
                totalSize = sizePos + sizeOther + sizeCol + sizeSHs + sizeChunk;
            }

            const float ksizeColWidth = 70;
            EditorGUI.BeginDisabledGroup(_quality != DataQuality.Custom);
            EditorGUI.indentLevel++;
            GUILayout.BeginHorizontal();
            _formatPos = (GaussianSplat.VectorFormat)EditorGUILayout.EnumPopup("Position", _formatPos);
            GUILayout.Label(sizePos > 0 ? EditorUtility.FormatBytes(sizePos) : string.Empty, GUILayout.Width(ksizeColWidth));
            GUILayout.EndHorizontal();
            GUILayout.BeginHorizontal();
            _formatScale = (GaussianSplat.VectorFormat)EditorGUILayout.EnumPopup("Scale", _formatScale);
            GUILayout.Label(sizeOther > 0 ? EditorUtility.FormatBytes(sizeOther) : string.Empty, GUILayout.Width(ksizeColWidth));
            GUILayout.EndHorizontal();
            GUILayout.BeginHorizontal();
            _formatColor = (GaussianSplat.ColorFormat)EditorGUILayout.EnumPopup("Color", _formatColor);
            GUILayout.Label(sizeCol > 0 ? EditorUtility.FormatBytes(sizeCol) : string.Empty, GUILayout.Width(ksizeColWidth));
            GUILayout.EndHorizontal();
            GUILayout.BeginHorizontal();
            _formatSH = (GaussianSplat.SHFormat) EditorGUILayout.EnumPopup("SH", _formatSH);
            GUIContent shGC = new GUIContent();
            shGC.text = sizeSHs > 0 ? EditorUtility.FormatBytes(sizeSHs) : string.Empty;
            if (_formatSH >= GaussianSplat.SHFormat.Cluster64K)
            {
                shGC.tooltip = "Note that SH clustering is not fast! (3-10 minutes for 6M splats)";
                shGC.image = EditorGUIUtility.IconContent("console.warnicon.sml").image;
            }
            GUILayout.Label(shGC, GUILayout.Width(ksizeColWidth));
            GUILayout.EndHorizontal();
            EditorGUI.indentLevel--;
            EditorGUI.EndDisabledGroup();
            if (totalSize > 0)
                EditorGUILayout.LabelField("Splat Size", $"{EditorUtility.FormatBytes(totalSize)} - {(double) _prevFileSize / totalSize:F2}x smaller");
            else
                GUILayout.Space(EditorGUIUtility.singleLineHeight);


            EditorGUILayout.Space();
            GUILayout.BeginHorizontal();
            GUILayout.Space(30);
            if (GUILayout.Button("Create Splat"))
            {
                CreateSplat();
            }
            GUILayout.Space(30);
            GUILayout.EndHorizontal();

            if (!string.IsNullOrWhiteSpace(_errorMessage))
            {
                EditorGUILayout.HelpBox(_errorMessage, MessageType.Error);
            }
        }

        private void ApplyQualityLevel()
        {
            switch (_quality)
            {
                case DataQuality.Custom:
                    break;
                case DataQuality.VeryLow: // 18.62x smaller, 32.27 PSNR
                    _formatPos = GaussianSplat.VectorFormat.Norm11;
                    _formatScale = GaussianSplat.VectorFormat.Norm6;
                    _formatColor = GaussianSplat.ColorFormat.BC7;
                    _formatSH = GaussianSplat.SHFormat.Cluster4K;
                    break;
                case DataQuality.Low: // 14.01x smaller, 35.17 PSNR
                    _formatPos = GaussianSplat.VectorFormat.Norm11;
                    _formatScale = GaussianSplat.VectorFormat.Norm6;
                    _formatColor = GaussianSplat.ColorFormat.Norm8X4;
                    _formatSH = GaussianSplat.SHFormat.Cluster16K;
                    break;
                case DataQuality.Medium: // 5.14x smaller, 47.46 PSNR
                    _formatPos = GaussianSplat.VectorFormat.Norm11;
                    _formatScale = GaussianSplat.VectorFormat.Norm11;
                    _formatColor = GaussianSplat.ColorFormat.Norm8X4;
                    _formatSH = GaussianSplat.SHFormat.Norm6;
                    break;
                case DataQuality.High: // 2.94x smaller, 57.77 PSNR
                    _formatPos = GaussianSplat.VectorFormat.Norm16;
                    _formatScale = GaussianSplat.VectorFormat.Norm16;
                    _formatColor = GaussianSplat.ColorFormat.Float16X4;
                    _formatSH = GaussianSplat.SHFormat.Norm11;
                    break;
                case DataQuality.VeryHigh: // 1.05x smaller
                    _formatPos = GaussianSplat.VectorFormat.Float32;
                    _formatScale = GaussianSplat.VectorFormat.Float32;
                    _formatColor = GaussianSplat.ColorFormat.Float32X4;
                    _formatSH = GaussianSplat.SHFormat.Float32;
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        // REFACTOR:CW this is going to have to create or deserialize
        private static T CreateOrReplaceSplat<T>(T splat, string path) where T : UnityEngine.Object
        {
            T result = AssetDatabase.LoadAssetAtPath<T>(path);
            if (result == null)
            {
                AssetDatabase.CreateAsset(splat, path);
                result = splat;
            }
            else
            {
                if (typeof(Mesh).IsAssignableFrom(typeof(T)))
                {
                    var mesh = result as Mesh;
                    if (mesh != null)
                    {
                        mesh.Clear();
                    }
                }
                EditorUtility.CopySerialized(splat, result);
            }
            return result;
        }

        private unsafe void CreateSplat()
        {
            _errorMessage = null;
            if (string.IsNullOrWhiteSpace(_inputFile))
            {
                _errorMessage = "Select input PLY/SPZ file";
                return;
            }

            if (string.IsNullOrWhiteSpace(_outputFolder))
            {
                _errorMessage = "Select output path";
                return;
            }
            Directory.CreateDirectory(_outputFolder);

            EditorUtility.DisplayProgressBar(kProgressTitle, "Reading data files", 0.0f);
            GaussianSplat.CameraInfo[] cameras = LoadJsonCamerasFile(_inputFile, _importCameras);
            using NativeArray<InputSplatData> inputSplats = LoadInputSplatFile(_inputFile);
            if (inputSplats.Length == 0)
            {
                EditorUtility.ClearProgressBar();
                return;
            }

            float3 boundsMin, boundsMax;
            var boundsJob = new CalcBoundsJob
            {
                BoundsMin = &boundsMin,
                BoundsMax = &boundsMax,
                SplatData = inputSplats
            };
            boundsJob.Schedule().Complete();

            EditorUtility.DisplayProgressBar(kProgressTitle, "Morton reordering", 0.05f);
            ReorderMorton(inputSplats, boundsMin, boundsMax);

            // cluster SHs
            NativeArray<int> splatSHIndices = default;
            NativeArray<GaussianSplat.SHTableItemFloat16> clusteredSHs = default;
            if (_formatSH >= GaussianSplat.SHFormat.Cluster64K)
            {
                EditorUtility.DisplayProgressBar(kProgressTitle, "Cluster SHs", 0.2f);
                ClusterSHs(inputSplats, _formatSH, out clusteredSHs, out splatSHIndices);
            }

            string baseName = Path.GetFileNameWithoutExtension(FilePickerControl.PathToDisplayString(_inputFile));

            EditorUtility.DisplayProgressBar(kProgressTitle, "Creating data objects", 0.7f);
            GaussianSplat splat = new GaussianSplat();
            splat.Initialize(inputSplats.Length, _formatPos, _formatScale, _formatColor, _formatSH, boundsMin, boundsMax, cameras);
            splat.Name = baseName;

            var dataHash = new Hash128((uint)splat.SplatCount, (uint)splat.FormatVersion, 0, 0);
            string pathChunk = $"{_outputFolder}/{baseName}_chk.bytes";
            string pathPos = $"{_outputFolder}/{baseName}_pos.bytes";
            string pathOther = $"{_outputFolder}/{baseName}_oth.bytes";
            string pathCol = $"{_outputFolder}/{baseName}_col.bytes";
            string pathSh = $"{_outputFolder}/{baseName}_shs.bytes";

            // if we are using full lossless (FP32) data, then do not use any chunking, and keep data as-is
            bool useChunks = IsUsingChunks;
            if (useChunks)
                CreateChunkData(inputSplats, pathChunk, ref dataHash);
            CreatePositionsData(inputSplats, pathPos, ref dataHash);
            CreateOtherData(inputSplats, pathOther, ref dataHash, splatSHIndices);
            CreateColorData(inputSplats, pathCol, ref dataHash);
            CreateSHData(inputSplats, pathSh, ref dataHash, clusteredSHs);
            splat.SetDataHash(dataHash);

            splatSHIndices.Dispose();
            clusteredSHs.Dispose();

            // REFACTOR:CW none of this asset code is going to work

            // files are created, import them so we can get to the imported objects, ugh
            EditorUtility.DisplayProgressBar(kProgressTitle, "Initial texture import", 0.85f);
            AssetDatabase.Refresh(ImportAssetOptions.ForceUncompressedImport);

            EditorUtility.DisplayProgressBar(kProgressTitle, "Setup data onto asset", 0.95f);
            splat.SetSplatFiles(
                useChunks ? AssetDatabase.LoadAssetAtPath<TextAsset>(pathChunk) : null,
                AssetDatabase.LoadAssetAtPath<TextAsset>(pathPos),
                AssetDatabase.LoadAssetAtPath<TextAsset>(pathOther),
                AssetDatabase.LoadAssetAtPath<TextAsset>(pathCol),
                AssetDatabase.LoadAssetAtPath<TextAsset>(pathSh));

            var splatPath = $"{_outputFolder}/{baseName}.splat";
            var savedSplat = CreateOrReplaceSplat(splat, splatPath);

            EditorUtility.DisplayProgressBar(kProgressTitle, "Saving splat", 0.99f);
            AssetDatabase.SaveAssets();
            EditorUtility.ClearProgressBar();

            Selection.activeObject = savedSplat;
        }

        private NativeArray<InputSplatData> LoadInputSplatFile(string filePath)
        {
            NativeArray<InputSplatData> data = default;
            if (!File.Exists(filePath))
            {
                _errorMessage = $"Did not find {filePath} file";
                return data;
            }
            try
            {
                GaussianFileReader.ReadFile(filePath, out data);
            }
            catch (Exception ex)
            {
                _errorMessage = ex.Message;
            }
            return data;
        }

        [BurstCompile]
        private struct CalcBoundsJob : IJob
        {
            [NativeDisableUnsafePtrRestriction] public unsafe float3* BoundsMin;
            [NativeDisableUnsafePtrRestriction] public unsafe float3* BoundsMax;
            [ReadOnly] public NativeArray<InputSplatData> SplatData;

            public unsafe void Execute()
            {
                float3 boundsMin = float.PositiveInfinity;
                float3 boundsMax = float.NegativeInfinity;

                for (int i = 0; i < SplatData.Length; ++i)
                {
                    float3 pos = SplatData[i].Pos;
                    boundsMin = math.min(boundsMin, pos);
                    boundsMax = math.max(boundsMax, pos);
                }
                *BoundsMin = boundsMin;
                *BoundsMax = boundsMax;
            }
        }

        [BurstCompile]
        private struct ReorderMortonJob : IJobParallelFor
        {
            private const float kScaler = (float) ((1 << 21) - 1);
            public float3 BoundsMin;
            public float3 InvBoundsSize;
            [ReadOnly] public NativeArray<InputSplatData> SplatData;
            public NativeArray<(ulong,int)> Order;

            public void Execute(int index)
            {
                float3 pos = ((float3)SplatData[index].Pos - BoundsMin) * InvBoundsSize * kScaler;
                uint3 iPos = (uint3) pos;
                ulong code = GaussianUtils.MortonEncode3(iPos);
                Order[index] = (code, index);
            }
        }

        private struct OrderComparer : IComparer<(ulong, int)> {
            public int Compare((ulong, int) a, (ulong, int) b)
            {
                if (a.Item1 < b.Item1) return -1;
                if (a.Item1 > b.Item1) return +1;
                return a.Item2 - b.Item2;
            }
        }

        private static void ReorderMorton(NativeArray<InputSplatData> splatData, float3 boundsMin, float3 boundsMax)
        {
            ReorderMortonJob order = new ReorderMortonJob
            {
                SplatData = splatData,
                BoundsMin = boundsMin,
                InvBoundsSize = 1.0f / (boundsMax - boundsMin),
                Order = new NativeArray<(ulong, int)>(splatData.Length, Allocator.TempJob)
            };
            order.Schedule(splatData.Length, 4096).Complete();
            order.Order.Sort(new OrderComparer());

            NativeArray<InputSplatData> copy = new(order.SplatData, Allocator.TempJob);
            for (int i = 0; i < copy.Length; ++i)
                order.SplatData[i] = copy[order.Order[i].Item2];
            copy.Dispose();

            order.Order.Dispose();
        }

        [BurstCompile]
        private static unsafe void GatherSHs(int splatCount, InputSplatData* splatData, float* shData)
        {
            for (int i = 0; i < splatCount; ++i)
            {
                UnsafeUtility.MemCpy(shData, ((float*)splatData) + 9, 15 * 3 * sizeof(float));
                splatData++;
                shData += 15 * 3;
            }
        }

        [BurstCompile]
        private struct ConvertSHClustersJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float3> Input;
            public NativeArray<GaussianSplat.SHTableItemFloat16> Output;
            public void Execute(int index)
            {
                var address = index * 15;
                GaussianSplat.SHTableItemFloat16 res;
                res.SH1 = new half3(Input[address+0]);
                res.SH2 = new half3(Input[address+1]);
                res.SH3 = new half3(Input[address+2]);
                res.SH4 = new half3(Input[address+3]);
                res.SH5 = new half3(Input[address+4]);
                res.SH6 = new half3(Input[address+5]);
                res.SH7 = new half3(Input[address+6]);
                res.SH8 = new half3(Input[address+7]);
                res.SH9 = new half3(Input[address+8]);
                res.SHA = new half3(Input[address+9]);
                res.SHB = new half3(Input[address+10]);
                res.SHC = new half3(Input[address+11]);
                res.SHD = new half3(Input[address+12]);
                res.SHE = new half3(Input[address+13]);
                res.SHF = new half3(Input[address+14]);
                res.SHPadding = default;
                Output[index] = res;
            }
        }

        private static bool ClusterSHProgress(float val)
        {
            EditorUtility.DisplayProgressBar(kProgressTitle, $"Cluster SHs ({val:P0})", 0.2f + val * 0.5f);
            return true;
        }

        private static unsafe void ClusterSHs(NativeArray<InputSplatData> splatData, GaussianSplat.SHFormat format, out NativeArray<GaussianSplat.SHTableItemFloat16> shs, out NativeArray<int> shIndices)
        {
            shs = default;
            shIndices = default;

            int shCount = GaussianSplat.GetSHCount(format, splatData.Length);
            if (shCount >= splatData.Length) // no need to cluster, just use raw data
                return;

            const int kshDim = 15 * 3;
            const int kbatchSize = 2048;
            float passesOverData = format switch
            {
                GaussianSplat.SHFormat.Cluster64K => 0.3f,
                GaussianSplat.SHFormat.Cluster32K => 0.4f,
                GaussianSplat.SHFormat.Cluster16K => 0.5f,
                GaussianSplat.SHFormat.Cluster8K => 0.8f,
                GaussianSplat.SHFormat.Cluster4K => 1.2f,
                _ => throw new ArgumentOutOfRangeException(nameof(format), format, null)
            };

            float t0 = Time.realtimeSinceStartup;
            NativeArray<float> shData = new(splatData.Length * kshDim, Allocator.Persistent);
            GatherSHs(splatData.Length, (InputSplatData*) splatData.GetUnsafeReadOnlyPtr(), (float*) shData.GetUnsafePtr());

            NativeArray<float> shMeans = new(shCount * kshDim, Allocator.Persistent);
            shIndices = new(splatData.Length, Allocator.Persistent);

            KMeansClustering.Calculate(kshDim, shData, kbatchSize, passesOverData, ClusterSHProgress, shMeans, shIndices);
            shData.Dispose();

            shs = new NativeArray<GaussianSplat.SHTableItemFloat16>(shCount, Allocator.Persistent);

            ConvertSHClustersJob job = new ConvertSHClustersJob
            {
                Input = shMeans.Reinterpret<float3>(4),
                Output = shs
            };
            job.Schedule(shCount, 256).Complete();
            shMeans.Dispose();
            float t1 = Time.realtimeSinceStartup;
            Debug.Log($"GS: clustered {splatData.Length/1000000.0:F2}M SHs into {shCount/1024}K ({passesOverData:F1}pass/{kbatchSize}batch) in {t1-t0:F0}s");
        }

        [BurstCompile]
        private struct CalcChunkDataJob : IJobParallelFor
        {
            [NativeDisableParallelForRestriction] public NativeArray<InputSplatData> SplatData;
            public NativeArray<GaussianSplat.ChunkInfo> Chunks;

            public void Execute(int chunkIdx)
            {
                float3 chunkMinPos = float.PositiveInfinity;
                float3 chunkMinScl = float.PositiveInfinity;
                float4 chunkMinCol = float.PositiveInfinity;
                float3 chunkMinShs = float.PositiveInfinity;
                float3 chunkMaxPos = float.NegativeInfinity;
                float3 chunkMaxScl = float.NegativeInfinity;
                float4 chunkMaxCol = float.NegativeInfinity;
                float3 chunkMaxShs = float.NegativeInfinity;

                int splatBegin = math.min(chunkIdx * GaussianSplat.kChunkSize, SplatData.Length);
                int splatEnd = math.min((chunkIdx + 1) * GaussianSplat.kChunkSize, SplatData.Length);

                // calculate data bounds inside the chunk
                for (int i = splatBegin; i < splatEnd; ++i)
                {
                    InputSplatData s = SplatData[i];

                    // transform scale to be more uniformly distributed
                    s.Scale = math.pow(s.Scale, 1.0f / 8.0f);
                    // transform opacity to be more uniformly distributed
                    s.Opacity = GaussianUtils.SquareCentered01(s.Opacity);
                    SplatData[i] = s;

                    chunkMinPos = math.min(chunkMinPos, s.Pos);
                    chunkMinScl = math.min(chunkMinScl, s.Scale);
                    chunkMinCol = math.min(chunkMinCol, new float4(s.Dc0, s.Opacity));
                    chunkMinShs = math.min(chunkMinShs, s.SH1);
                    chunkMinShs = math.min(chunkMinShs, s.SH2);
                    chunkMinShs = math.min(chunkMinShs, s.SH3);
                    chunkMinShs = math.min(chunkMinShs, s.SH4);
                    chunkMinShs = math.min(chunkMinShs, s.SH5);
                    chunkMinShs = math.min(chunkMinShs, s.SH6);
                    chunkMinShs = math.min(chunkMinShs, s.SH7);
                    chunkMinShs = math.min(chunkMinShs, s.SH8);
                    chunkMinShs = math.min(chunkMinShs, s.SH9);
                    chunkMinShs = math.min(chunkMinShs, s.SHA);
                    chunkMinShs = math.min(chunkMinShs, s.SHB);
                    chunkMinShs = math.min(chunkMinShs, s.SHC);
                    chunkMinShs = math.min(chunkMinShs, s.SHD);
                    chunkMinShs = math.min(chunkMinShs, s.SHE);
                    chunkMinShs = math.min(chunkMinShs, s.SHF);

                    chunkMaxPos = math.max(chunkMaxPos, s.Pos);
                    chunkMaxScl = math.max(chunkMaxScl, s.Scale);
                    chunkMaxCol = math.max(chunkMaxCol, new float4(s.Dc0, s.Opacity));
                    chunkMaxShs = math.max(chunkMaxShs, s.SH1);
                    chunkMaxShs = math.max(chunkMaxShs, s.SH2);
                    chunkMaxShs = math.max(chunkMaxShs, s.SH3);
                    chunkMaxShs = math.max(chunkMaxShs, s.SH4);
                    chunkMaxShs = math.max(chunkMaxShs, s.SH5);
                    chunkMaxShs = math.max(chunkMaxShs, s.SH6);
                    chunkMaxShs = math.max(chunkMaxShs, s.SH7);
                    chunkMaxShs = math.max(chunkMaxShs, s.SH8);
                    chunkMaxShs = math.max(chunkMaxShs, s.SH9);
                    chunkMaxShs = math.max(chunkMaxShs, s.SHA);
                    chunkMaxShs = math.max(chunkMaxShs, s.SHB);
                    chunkMaxShs = math.max(chunkMaxShs, s.SHC);
                    chunkMaxShs = math.max(chunkMaxShs, s.SHD);
                    chunkMaxShs = math.max(chunkMaxShs, s.SHE);
                    chunkMaxShs = math.max(chunkMaxShs, s.SHF);
                }

                // make sure bounds are not zero
                chunkMaxPos = math.max(chunkMaxPos, chunkMinPos + 1.0e-5f);
                chunkMaxScl = math.max(chunkMaxScl, chunkMinScl + 1.0e-5f);
                chunkMaxCol = math.max(chunkMaxCol, chunkMinCol + 1.0e-5f);
                chunkMaxShs = math.max(chunkMaxShs, chunkMinShs + 1.0e-5f);

                // store chunk info
                GaussianSplat.ChunkInfo info = default;
                info.PosX = new float2(chunkMinPos.x, chunkMaxPos.x);
                info.PosY = new float2(chunkMinPos.y, chunkMaxPos.y);
                info.PosZ = new float2(chunkMinPos.z, chunkMaxPos.z);
                info.SclX = math.f32tof16(chunkMinScl.x) | (math.f32tof16(chunkMaxScl.x) << 16);
                info.SclY = math.f32tof16(chunkMinScl.y) | (math.f32tof16(chunkMaxScl.y) << 16);
                info.SclZ = math.f32tof16(chunkMinScl.z) | (math.f32tof16(chunkMaxScl.z) << 16);
                info.ColR = math.f32tof16(chunkMinCol.x) | (math.f32tof16(chunkMaxCol.x) << 16);
                info.ColG = math.f32tof16(chunkMinCol.y) | (math.f32tof16(chunkMaxCol.y) << 16);
                info.ColB = math.f32tof16(chunkMinCol.z) | (math.f32tof16(chunkMaxCol.z) << 16);
                info.ColA = math.f32tof16(chunkMinCol.w) | (math.f32tof16(chunkMaxCol.w) << 16);
                info.SHR = math.f32tof16(chunkMinShs.x) | (math.f32tof16(chunkMaxShs.x) << 16);
                info.SHG = math.f32tof16(chunkMinShs.y) | (math.f32tof16(chunkMaxShs.y) << 16);
                info.SHB = math.f32tof16(chunkMinShs.z) | (math.f32tof16(chunkMaxShs.z) << 16);
                Chunks[chunkIdx] = info;

                // adjust data to be 0..1 within chunk bounds
                for (int i = splatBegin; i < splatEnd; ++i)
                {
                    InputSplatData s = SplatData[i];
                    s.Pos = ((float3)s.Pos - chunkMinPos) / (chunkMaxPos - chunkMinPos);
                    s.Scale = ((float3)s.Scale - chunkMinScl) / (chunkMaxScl - chunkMinScl);
                    s.Dc0 = ((float3)s.Dc0 - chunkMinCol.xyz) / (chunkMaxCol.xyz - chunkMinCol.xyz);
                    s.Opacity = (s.Opacity - chunkMinCol.w) / (chunkMaxCol.w - chunkMinCol.w);
                    s.SH1 = ((float3) s.SH1 - chunkMinShs) / (chunkMaxShs - chunkMinShs);
                    s.SH2 = ((float3) s.SH2 - chunkMinShs) / (chunkMaxShs - chunkMinShs);
                    s.SH3 = ((float3) s.SH3 - chunkMinShs) / (chunkMaxShs - chunkMinShs);
                    s.SH4 = ((float3) s.SH4 - chunkMinShs) / (chunkMaxShs - chunkMinShs);
                    s.SH5 = ((float3) s.SH5 - chunkMinShs) / (chunkMaxShs - chunkMinShs);
                    s.SH6 = ((float3) s.SH6 - chunkMinShs) / (chunkMaxShs - chunkMinShs);
                    s.SH7 = ((float3) s.SH7 - chunkMinShs) / (chunkMaxShs - chunkMinShs);
                    s.SH8 = ((float3) s.SH8 - chunkMinShs) / (chunkMaxShs - chunkMinShs);
                    s.SH9 = ((float3) s.SH9 - chunkMinShs) / (chunkMaxShs - chunkMinShs);
                    s.SHA = ((float3) s.SHA - chunkMinShs) / (chunkMaxShs - chunkMinShs);
                    s.SHB = ((float3) s.SHB - chunkMinShs) / (chunkMaxShs - chunkMinShs);
                    s.SHC = ((float3) s.SHC - chunkMinShs) / (chunkMaxShs - chunkMinShs);
                    s.SHD = ((float3) s.SHD - chunkMinShs) / (chunkMaxShs - chunkMinShs);
                    s.SHE = ((float3) s.SHE - chunkMinShs) / (chunkMaxShs - chunkMinShs);
                    s.SHF = ((float3) s.SHF - chunkMinShs) / (chunkMaxShs - chunkMinShs);
                    SplatData[i] = s;
                }
            }
        }

        private static void CreateChunkData(NativeArray<InputSplatData> splatData, string filePath, ref Hash128 dataHash)
        {
            int chunkCount = (splatData.Length + GaussianSplat.kChunkSize - 1) / GaussianSplat.kChunkSize;
            CalcChunkDataJob job = new CalcChunkDataJob
            {
                SplatData = splatData,
                Chunks = new(chunkCount, Allocator.TempJob),
            };

            job.Schedule(chunkCount, 8).Complete();

            dataHash.Append(ref job.Chunks);

            using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write);
            fs.Write(job.Chunks.Reinterpret<byte>(UnsafeUtility.SizeOf<GaussianSplat.ChunkInfo>()));

            job.Chunks.Dispose();
        }

        [BurstCompile]
        private struct ConvertColorJob : IJobParallelFor
        {
            public int Width;
            public int Height;
            [ReadOnly] public NativeArray<float4> InputData;
            [NativeDisableParallelForRestriction] public NativeArray<byte> OutputData;
            public GaussianSplat.ColorFormat Format;
            public int FormatBytesPerPixel;

            public unsafe void Execute(int y)
            {
                int srcIdx = y * Width;
                byte* dstPtr = (byte*) OutputData.GetUnsafePtr() + y * Width * FormatBytesPerPixel;
                for (int x = 0; x < Width; ++x)
                {
                    float4 pix = InputData[srcIdx];

                    switch (Format)
                    {
                        case GaussianSplat.ColorFormat.Float32X4:
                        {
                            *(float4*) dstPtr = pix;
                        }
                            break;
                        case GaussianSplat.ColorFormat.Float16X4:
                        {
                            half4 enc = new half4(pix);
                            *(half4*) dstPtr = enc;
                        }
                            break;
                        case GaussianSplat.ColorFormat.Norm8X4:
                        {
                            pix = math.saturate(pix);
                            uint enc = (uint)(pix.x * 255.5f) | ((uint)(pix.y * 255.5f) << 8) | ((uint)(pix.z * 255.5f) << 16) | ((uint)(pix.w * 255.5f) << 24);
                            *(uint*) dstPtr = enc;
                        }
                            break;
                    }

                    srcIdx++;
                    dstPtr += FormatBytesPerPixel;
                }
            }
        }

        private static ulong EncodeFloat3ToNorm16(float3 v) // 48 bits: 16.16.16
        {
            return (ulong) (v.x * 65535.5f) | ((ulong) (v.y * 65535.5f) << 16) | ((ulong) (v.z * 65535.5f) << 32);
        }

        private static uint EncodeFloat3ToNorm11(float3 v) // 32 bits: 11.10.11
        {
            return (uint) (v.x * 2047.5f) | ((uint) (v.y * 1023.5f) << 11) | ((uint) (v.z * 2047.5f) << 21);
        }

        private static ushort EncodeFloat3ToNorm655(float3 v) // 16 bits: 6.5.5
        {
            return (ushort) ((uint) (v.x * 63.5f) | ((uint) (v.y * 31.5f) << 6) | ((uint) (v.z * 31.5f) << 11));
        }

        private static ushort EncodeFloat3ToNorm565(float3 v) // 16 bits: 5.6.5
        {
            return (ushort) ((uint) (v.x * 31.5f) | ((uint) (v.y * 63.5f) << 5) | ((uint) (v.z * 31.5f) << 11));
        }

        private static uint EncodeQuaternionToNorm10(float4 v) // 32 bits: 10.10.10.2
        {
            return (uint) (v.x * 1023.5f) | ((uint) (v.y * 1023.5f) << 10) | ((uint) (v.z * 1023.5f) << 20) | ((uint) (v.w * 3.5f) << 30);
        }

        private static unsafe void EmitEncodedVector(float3 v, byte* outputPtr, GaussianSplat.VectorFormat format)
        {
            switch (format)
            {
                case GaussianSplat.VectorFormat.Float32:
                {
                    *(float*) outputPtr = v.x;
                    *(float*) (outputPtr + 4) = v.y;
                    *(float*) (outputPtr + 8) = v.z;
                }
                    break;
                case GaussianSplat.VectorFormat.Norm16:
                {
                    ulong enc = EncodeFloat3ToNorm16(math.saturate(v));
                    *(uint*) outputPtr = (uint) enc;
                    *(ushort*) (outputPtr + 4) = (ushort) (enc >> 32);
                }
                    break;
                case GaussianSplat.VectorFormat.Norm11:
                {
                    uint enc = EncodeFloat3ToNorm11(math.saturate(v));
                    *(uint*) outputPtr = enc;
                }
                    break;
                case GaussianSplat.VectorFormat.Norm6:
                {
                    ushort enc = EncodeFloat3ToNorm655(math.saturate(v));
                    *(ushort*) outputPtr = enc;
                }
                    break;
            }
        }

        [BurstCompile]
        private struct CreatePositionsDataJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<InputSplatData> MInput;
            public GaussianSplat.VectorFormat Format;
            public int FormatSize;
            [NativeDisableParallelForRestriction] public NativeArray<byte> Output;

            public unsafe void Execute(int index)
            {
                byte* outputPtr = (byte*) Output.GetUnsafePtr() + index * FormatSize;
                EmitEncodedVector(MInput[index].Pos, outputPtr, Format);
            }
        }

        [BurstCompile]
        private struct CreateOtherDataJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<InputSplatData> Input;
            [NativeDisableContainerSafetyRestriction] [ReadOnly] public NativeArray<int> SplatSHIndices;
            public GaussianSplat.VectorFormat ScaleFormat;
            public int FormatSize;
            [NativeDisableParallelForRestriction] public NativeArray<byte> Output;

            public unsafe void Execute(int index)
            {
                byte* outputPtr = (byte*) Output.GetUnsafePtr() + index * FormatSize;

                // rotation: 4 bytes
                {
                    Quaternion rotQ = Input[index].Rot;
                    float4 rot = new float4(rotQ.x, rotQ.y, rotQ.z, rotQ.w);
                    uint enc = EncodeQuaternionToNorm10(rot);
                    *(uint*) outputPtr = enc;
                    outputPtr += 4;
                }

                // scale: 6, 4 or 2 bytes
                EmitEncodedVector(Input[index].Scale, outputPtr, ScaleFormat);
                outputPtr += GaussianSplat.GetVectorSize(ScaleFormat);

                // SH index
                if (SplatSHIndices.IsCreated)
                    *(ushort*) outputPtr = (ushort)SplatSHIndices[index];
            }
        }

        private static int NextMultipleOf(int size, int multipleOf)
        {
            return (size + multipleOf - 1) / multipleOf * multipleOf;
        }

        private void CreatePositionsData(NativeArray<InputSplatData> inputSplats, string filePath, ref Hash128 dataHash)
        {
            int dataLen = inputSplats.Length * GaussianSplat.GetVectorSize(_formatPos);
            dataLen = NextMultipleOf(dataLen, 8); // serialized as ulong
            NativeArray<byte> data = new(dataLen, Allocator.TempJob);

            CreatePositionsDataJob job = new CreatePositionsDataJob
            {
                MInput = inputSplats,
                Format = _formatPos,
                FormatSize = GaussianSplat.GetVectorSize(_formatPos),
                Output = data
            };
            job.Schedule(inputSplats.Length, 8192).Complete();

            dataHash.Append(data);

            using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write);
            fs.Write(data);

            data.Dispose();
        }

        private void CreateOtherData(NativeArray<InputSplatData> inputSplats, string filePath, ref Hash128 dataHash, NativeArray<int> splatSHIndices)
        {
            int formatSize = GaussianSplat.GetOtherSizeNoSHIndex(_formatScale);
            if (splatSHIndices.IsCreated)
                formatSize += 2;
            int dataLen = inputSplats.Length * formatSize;

            dataLen = NextMultipleOf(dataLen, 8); // serialized as ulong
            NativeArray<byte> data = new(dataLen, Allocator.TempJob);

            CreateOtherDataJob job = new CreateOtherDataJob
            {
                Input = inputSplats,
                SplatSHIndices = splatSHIndices,
                ScaleFormat = _formatScale,
                FormatSize = formatSize,
                Output = data
            };
            job.Schedule(inputSplats.Length, 8192).Complete();

            dataHash.Append(data);

            using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write);
            fs.Write(data);

            data.Dispose();
        }

        private static int SplatIndexToTextureIndex(uint idx)
        {
            uint2 xy = GaussianUtils.DecodeMorton2D_16x16(idx);
            uint width = GaussianSplat.kTextureWidth / 16;
            idx >>= 8;
            uint x = (idx % width) * 16 + xy.x;
            uint y = (idx / width) * 16 + xy.y;
            return (int)(y * GaussianSplat.kTextureWidth + x);
        }

        [BurstCompile]
        private struct CreateColorDataJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<InputSplatData> Input;
            [NativeDisableParallelForRestriction] public NativeArray<float4> Output;

            public void Execute(int index)
            {
                var splat = Input[index];
                int i = SplatIndexToTextureIndex((uint)index);
                Output[i] = new float4(splat.Dc0.x, splat.Dc0.y, splat.Dc0.z, splat.Opacity);
            }
        }

        private void CreateColorData(NativeArray<InputSplatData> inputSplats, string filePath, ref Hash128 dataHash)
        {
            var (width, height) = GaussianSplat.CalcTextureSize(inputSplats.Length);
            NativeArray<float4> data = new(width * height, Allocator.TempJob);

            CreateColorDataJob job = new CreateColorDataJob();
            job.Input = inputSplats;
            job.Output = data;
            job.Schedule(inputSplats.Length, 8192).Complete();

            dataHash.Append(data);
            dataHash.Append((int)_formatColor);

            GraphicsFormat gfxFormat = GaussianSplat.ColorFormatToGraphics(_formatColor);
            int dstSize = (int)GraphicsFormatUtility.ComputeMipmapSize(width, height, gfxFormat);

            if (GraphicsFormatUtility.IsCompressedFormat(gfxFormat))
            {
                Texture2D tex = new Texture2D(width, height, GraphicsFormat.R32G32B32A32_SFloat, TextureCreationFlags.DontInitializePixels | TextureCreationFlags.DontUploadUponCreate);
                tex.SetPixelData(data, 0);
                EditorUtility.CompressTexture(tex, GraphicsFormatUtility.GetTextureFormat(gfxFormat), 100);
                NativeArray<byte> cmpData = tex.GetPixelData<byte>(0);
                using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write);
                fs.Write(cmpData);

                DestroyImmediate(tex);
            }
            else
            {
                ConvertColorJob jobConvert = new ConvertColorJob
                {
                    Width = width,
                    Height = height,
                    InputData = data,
                    Format = _formatColor,
                    OutputData = new NativeArray<byte>(dstSize, Allocator.TempJob),
                    FormatBytesPerPixel = dstSize / width / height
                };
                jobConvert.Schedule(height, 1).Complete();
                using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write);
                fs.Write(jobConvert.OutputData);
                jobConvert.OutputData.Dispose();
            }

            data.Dispose();
        }

        [BurstCompile]
        private struct CreateSHDataJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<InputSplatData> Input;
            public GaussianSplat.SHFormat Format;
            public NativeArray<byte> Output;
            public unsafe void Execute(int index)
            {
                var splat = Input[index];

                switch (Format)
                {
                    case GaussianSplat.SHFormat.Float32:
                    {
                        GaussianSplat.SHTableItemFloat32 res;
                        res.SH1 = splat.SH1;
                        res.SH2 = splat.SH2;
                        res.SH3 = splat.SH3;
                        res.SH4 = splat.SH4;
                        res.SH5 = splat.SH5;
                        res.SH6 = splat.SH6;
                        res.SH7 = splat.SH7;
                        res.SH8 = splat.SH8;
                        res.SH9 = splat.SH9;
                        res.SHA = splat.SHA;
                        res.SHB = splat.SHB;
                        res.SHC = splat.SHC;
                        res.SHD = splat.SHD;
                        res.SHE = splat.SHE;
                        res.SHF = splat.SHF;
                        res.SHPadding = default;
                        ((GaussianSplat.SHTableItemFloat32*) Output.GetUnsafePtr())[index] = res;
                    }
                        break;
                    case GaussianSplat.SHFormat.Float16:
                    {
                        GaussianSplat.SHTableItemFloat16 res;
                        res.SH1 = new half3(splat.SH1);
                        res.SH2 = new half3(splat.SH2);
                        res.SH3 = new half3(splat.SH3);
                        res.SH4 = new half3(splat.SH4);
                        res.SH5 = new half3(splat.SH5);
                        res.SH6 = new half3(splat.SH6);
                        res.SH7 = new half3(splat.SH7);
                        res.SH8 = new half3(splat.SH8);
                        res.SH9 = new half3(splat.SH9);
                        res.SHA = new half3(splat.SHA);
                        res.SHB = new half3(splat.SHB);
                        res.SHC = new half3(splat.SHC);
                        res.SHD = new half3(splat.SHD);
                        res.SHE = new half3(splat.SHE);
                        res.SHF = new half3(splat.SHF);
                        res.SHPadding = default;
                        ((GaussianSplat.SHTableItemFloat16*) Output.GetUnsafePtr())[index] = res;
                    }
                        break;
                    case GaussianSplat.SHFormat.Norm11:
                    {
                        GaussianSplat.SHTableItemNorm11 res;
                        res.SH1 = EncodeFloat3ToNorm11(splat.SH1);
                        res.SH2 = EncodeFloat3ToNorm11(splat.SH2);
                        res.SH3 = EncodeFloat3ToNorm11(splat.SH3);
                        res.SH4 = EncodeFloat3ToNorm11(splat.SH4);
                        res.SH5 = EncodeFloat3ToNorm11(splat.SH5);
                        res.SH6 = EncodeFloat3ToNorm11(splat.SH6);
                        res.SH7 = EncodeFloat3ToNorm11(splat.SH7);
                        res.SH8 = EncodeFloat3ToNorm11(splat.SH8);
                        res.SH9 = EncodeFloat3ToNorm11(splat.SH9);
                        res.SHA = EncodeFloat3ToNorm11(splat.SHA);
                        res.SHB = EncodeFloat3ToNorm11(splat.SHB);
                        res.SHC = EncodeFloat3ToNorm11(splat.SHC);
                        res.SHD = EncodeFloat3ToNorm11(splat.SHD);
                        res.SHE = EncodeFloat3ToNorm11(splat.SHE);
                        res.SHF = EncodeFloat3ToNorm11(splat.SHF);
                        ((GaussianSplat.SHTableItemNorm11*) Output.GetUnsafePtr())[index] = res;
                    }
                        break;
                    case GaussianSplat.SHFormat.Norm6:
                    {
                        GaussianSplat.SHTableItemNorm6 res;
                        res.SH1 = EncodeFloat3ToNorm565(splat.SH1);
                        res.SH2 = EncodeFloat3ToNorm565(splat.SH2);
                        res.SH3 = EncodeFloat3ToNorm565(splat.SH3);
                        res.SH4 = EncodeFloat3ToNorm565(splat.SH4);
                        res.SH5 = EncodeFloat3ToNorm565(splat.SH5);
                        res.SH6 = EncodeFloat3ToNorm565(splat.SH6);
                        res.SH7 = EncodeFloat3ToNorm565(splat.SH7);
                        res.SH8 = EncodeFloat3ToNorm565(splat.SH8);
                        res.SH9 = EncodeFloat3ToNorm565(splat.SH9);
                        res.SHA = EncodeFloat3ToNorm565(splat.SHA);
                        res.SHB = EncodeFloat3ToNorm565(splat.SHB);
                        res.SHC = EncodeFloat3ToNorm565(splat.SHC);
                        res.SHD = EncodeFloat3ToNorm565(splat.SHD);
                        res.SHE = EncodeFloat3ToNorm565(splat.SHE);
                        res.SHF = EncodeFloat3ToNorm565(splat.SHF);
                        res.SHPadding = 0;
                        ((GaussianSplat.SHTableItemNorm6*) Output.GetUnsafePtr())[index] = res;
                    }
                        break;
                }
            }
        }

        private static void EmitSimpleDataFile<T>(NativeArray<T> data, string filePath, ref Hash128 dataHash) where T : unmanaged
        {
            dataHash.Append(data);
            using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write);
            fs.Write(data.Reinterpret<byte>(UnsafeUtility.SizeOf<T>()));
        }

        private void CreateSHData(NativeArray<InputSplatData> inputSplats, string filePath, ref Hash128 dataHash, NativeArray<GaussianSplat.SHTableItemFloat16> clusteredSHs)
        {
            if (clusteredSHs.IsCreated)
            {
                EmitSimpleDataFile(clusteredSHs, filePath, ref dataHash);
            }
            else
            {
                int dataLen = (int)GaussianSplat.CalcSHDataSize(inputSplats.Length, _formatSH);
                NativeArray<byte> data = new(dataLen, Allocator.TempJob);
                CreateSHDataJob job = new CreateSHDataJob
                {
                    Input = inputSplats,
                    Format = _formatSH,
                    Output = data
                };
                job.Schedule(inputSplats.Length, 8192).Complete();
                EmitSimpleDataFile(data, filePath, ref dataHash);
                data.Dispose();
            }
        }

        private static GaussianSplat.CameraInfo[] LoadJsonCamerasFile(string curPath, bool doImport)
        {
            if (!doImport)
                return null;

            string camerasPath;
            while (true)
            {
                var dir = Path.GetDirectoryName(curPath);
                if (!Directory.Exists(dir))
                    return null;
                camerasPath = $"{dir}/{kCamerasJson}";
                if (File.Exists(camerasPath))
                    break;
                curPath = dir;
            }

            if (!File.Exists(camerasPath))
                return null;

            string json = File.ReadAllText(camerasPath);
            var jsonCameras = json.FromJson<List<JsonCamera>>();
            if (jsonCameras == null || jsonCameras.Count == 0)
                return null;

            var result = new GaussianSplat.CameraInfo[jsonCameras.Count];
            for (var camIndex = 0; camIndex < jsonCameras.Count; camIndex++)
            {
                var jsonCam = jsonCameras[camIndex];
                var pos = new Vector3(jsonCam.Position[0], jsonCam.Position[1], jsonCam.Position[2]);
                // the matrix is a "view matrix", not "camera matrix" lol
                var axisX = new Vector3(jsonCam.Rotation[0][0], jsonCam.Rotation[1][0], jsonCam.Rotation[2][0]);
                var axisY = new Vector3(jsonCam.Rotation[0][1], jsonCam.Rotation[1][1], jsonCam.Rotation[2][1]);
                var axisZ = new Vector3(jsonCam.Rotation[0][2], jsonCam.Rotation[1][2], jsonCam.Rotation[2][2]);

                axisY *= -1;
                axisZ *= -1;

                var cam = new GaussianSplat.CameraInfo
                {
                    Pos = pos,
                    AxisX = axisX,
                    AxisY = axisY,
                    AxisZ = axisZ,
                    FOV = 25 //@TODO
                };
                result[camIndex] = cam;
            }

            return result;
        }

        [Serializable]
        public class JsonCamera
        {
            public int ID;
            public string ImgName;
            public int Width;
            public int Height;
            public float[] Position;
            public float[][] Rotation;
            public float FX;
            public float Fy;
        }
    }
}
