/*
// SPDX-License-Identifier: MIT

using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace GaussianSplatting.Editor
{
    [BurstCompile]
    public static class GaussianSplatValidator
    {
        private struct RefItem
        {
            public string SplatPath;
            public int CameraIndex;
            public float FOV;
        }

        // currently on RTX 3080Ti: 43.76, 39.36, 43.50 PSNR
        [MenuItem("Tools/Gaussian Splats/Debug/Validate Render against SBIR")]
        public static void ValidateSBIR()
        {
            ValidateImpl("SBIR");
        }
        // currently on RTX 3080Ti: matches
        [MenuItem("Tools/Gaussian Splats/Debug/Validate Render against D3D12")]
        public static void ValidateD3D12()
        {
            ValidateImpl("D3D12");
        }

        private static unsafe void ValidateImpl(string refPrefix)
        {
            var renderer = Object.FindFirstObjectByType(typeof(GaussianSplatRenderer)) as GaussianSplatRenderer;
            {
                if (renderer == null)
                {
                    Debug.LogError("No GaussianSplatRenderer object found");
                    return;
                }
            }
            var items = new RefItem[]
            {
                new() {SplatPath = "bicycle", CameraIndex = 0, FOV = 39.09651f},
                new() {SplatPath = "truck", CameraIndex = 30, FOV = 50},
                new() {SplatPath = "garden", CameraIndex = 30, FOV = 47},
            };

            var cam = Camera.main;
            var oldSplat = renderer.Splat;
            var oldCamPos = cam.transform.localPosition;
            var oldCamRot = cam.transform.localRotation;
            var oldCamFov = cam.fieldOfView;

            for (var index = 0; index < items.Length; index++)
            {
                var item = items[index];
                EditorUtility.DisplayProgressBar("Validating Gaussian splat rendering", item.SplatPath, (float)index / items.Length);
                var path = $"Assets/GaussianAssets/{item.SplatPath}-point_cloud-iteration_30000-point_cloud.asset";
                var gs = AssetDatabase.LoadAssetAtPath<GaussianSplat>(path);
                if (gs == null)
                {
                    Debug.LogError($"Did not find splat for validation item {item.SplatPath} at {path}");
                    continue;
                }
                var refImageFile = $"../../docs/RefImages/{refPrefix}_{item.SplatPath}{item.CameraIndex}.png"; // use our snapshot by default
                if (!File.Exists(refImageFile))
                {
                    Debug.LogError($"Did not find reference image for validation item {item.SplatPath} at {refImageFile}");
                    continue;
                }

                var compareTexture = new Texture2D(4, 4, GraphicsFormat.R8G8B8A8_SRGB, TextureCreationFlags.None);
                byte[] refImageBytes = File.ReadAllBytes(refImageFile);
                ImageConversion.LoadImage(compareTexture, refImageBytes, false);

                int width = compareTexture.width;
                int height = compareTexture.height;

                var renderTarget = RenderTexture.GetTemporary(width, height, 24, GraphicsFormat.R8G8B8A8_SRGB);
                cam.targetTexture = renderTarget;
                cam.fieldOfView = item.FOV;

                var captureTexture = new Texture2D(width, height, GraphicsFormat.R8G8B8A8_SRGB, TextureCreationFlags.None);
                NativeArray<Color32> diffPixels = new(width * height, Allocator.Persistent);

                renderer.Splat = gs;
                renderer.Update();
                renderer.ActivateCamera(item.CameraIndex);
                cam.Render();
                Graphics.SetRenderTarget(renderTarget);
                captureTexture.ReadPixels(new Rect(0, 0, width, height), 0, 0);

                NativeArray<Color32> refPixels = compareTexture.GetPixelData<Color32>(0);
                NativeArray<Color32> gotPixels = captureTexture.GetPixelData<Color32>(0);
                float psnr = 0, rmse = 0;
                int errorsCount = 0;
                DiffImagesJob difJob = new DiffImagesJob();
                difJob.DifPixels = diffPixels;
                difJob.RefPixels = refPixels;
                difJob.GotPixels = gotPixels;
                difJob.PsnrPtr = &psnr;
                difJob.RmsePtr = &rmse;
                difJob.DifPixCount = &errorsCount;
                difJob.Schedule().Complete();

                string pathDif = $"../../Shot-{refPrefix}-{item.SplatPath}{item.CameraIndex}-diff.png";
                string pathRef = $"../../Shot-{refPrefix}-{item.SplatPath}{item.CameraIndex}-ref.png";
                string pathGot = $"../../Shot-{refPrefix}-{item.SplatPath}{item.CameraIndex}-got.png";

                if (errorsCount > 50 || psnr < 90.0f)
                {
                    Debug.LogWarning(
                        $"{refPrefix} {item.SplatPath} cam {item.CameraIndex}: RMSE {rmse:F2} PSNR {psnr:F2} diff pixels {errorsCount:N0}");

                    NativeArray<byte> pngBytes = ImageConversion.EncodeNativeArrayToPNG(diffPixels,
                        GraphicsFormat.R8G8B8A8_SRGB, (uint) width, (uint) height);
                    File.WriteAllBytes(pathDif, pngBytes.ToArray());
                    pngBytes.Dispose();
                    pngBytes = ImageConversion.EncodeNativeArrayToPNG(refPixels, GraphicsFormat.R8G8B8A8_SRGB,
                        (uint) width, (uint) height);
                    File.WriteAllBytes(pathRef, pngBytes.ToArray());
                    pngBytes.Dispose();
                    pngBytes = ImageConversion.EncodeNativeArrayToPNG(gotPixels, GraphicsFormat.R8G8B8A8_SRGB,
                        (uint) width, (uint) height);
                    File.WriteAllBytes(pathGot, pngBytes.ToArray());
                    pngBytes.Dispose();
                }
                else
                {
                    File.Delete(pathDif);
                    File.Delete(pathRef);
                    File.Delete(pathGot);
                }

                diffPixels.Dispose();
                RenderTexture.ReleaseTemporary(renderTarget);
                Object.DestroyImmediate(captureTexture);
                Object.DestroyImmediate(compareTexture);
            }

            cam.targetTexture = null;
            renderer.Splat = oldSplat;
            renderer.Update();
            cam.transform.localPosition = oldCamPos;
            cam.transform.localRotation = oldCamRot;
            cam.fieldOfView = oldCamFov;

            EditorUtility.ClearProgressBar();
        }

        [BurstCompile]
        private struct DiffImagesJob : IJob
        {
            public NativeArray<Color32> RefPixels;
            public NativeArray<Color32> GotPixels;
            public NativeArray<Color32> DifPixels;
            [NativeDisableUnsafePtrRestriction] public unsafe float* RmsePtr;
            [NativeDisableUnsafePtrRestriction] public unsafe float* PsnrPtr;
            [NativeDisableUnsafePtrRestriction] public unsafe int* DifPixCount;

            public unsafe void Execute()
            {
                const int kdiffScale = 5;
                const int kdiffThreshold = 3 * kdiffScale;
                *DifPixCount = 0;
                double sumSqDif = 0;
                for (int i = 0; i < RefPixels.Length; ++i)
                {
                    Color32 cref = RefPixels[i];
                    // note: LoadImage always loads PNGs into ARGB order, so swizzle to normal RGBA
                    cref = new Color32(cref.g, cref.b, cref.a, 255);
                    RefPixels[i] = cref;

                    Color32 cgot = GotPixels[i];
                    cgot.a = 255;
                    GotPixels[i] = cgot;

                    Color32 cdif = new Color32(0, 0, 0, 255);
                    cdif.r = (byte)math.abs(cref.r - cgot.r);
                    cdif.g = (byte)math.abs(cref.g - cgot.g);
                    cdif.b = (byte)math.abs(cref.b - cgot.b);
                    sumSqDif += cdif.r * cdif.r + cdif.g * cdif.g + cdif.b * cdif.b;

                    cdif.r = (byte)math.min(255, cdif.r * kdiffScale);
                    cdif.g = (byte)math.min(255, cdif.g * kdiffScale);
                    cdif.b = (byte)math.min(255, cdif.b * kdiffScale);
                    DifPixels[i] = cdif;
                    if (cdif.r >= kdiffThreshold || cdif.g >= kdiffThreshold || cdif.b >= kdiffThreshold)
                    {
                        (*DifPixCount)++;
                    }
                }

                double meanSqDif = sumSqDif / (RefPixels.Length * 3);
                double rmse = math.sqrt(meanSqDif);
                double psnr = 20.0 * math.log10(255.0) - 10.0 * math.log10(rmse * rmse);
                *RmsePtr = (float) rmse;
                *PsnrPtr = (float) psnr;
            }
        }
    }
}
*/