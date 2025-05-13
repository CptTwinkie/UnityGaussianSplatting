// SPDX-License-Identifier: MIT

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using GaussianSplatting.Runtime;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEditor;
using UnityEditor.EditorTools;
using UnityEngine;

namespace GaussianSplatting.Editor
{
    [CustomEditor(typeof(GaussianSplatRenderer))]
    [CanEditMultipleObjects]
    public class GaussianSplatRendererEditor : UnityEditor.Editor
    {
        private const string kPrefExportBake = "nesnausk.GaussianSplatting.ExportBakeTransform";

        private SerializedProperty _propSplat;
        private SerializedProperty _propRenderOrder;
        private SerializedProperty _propSplatScale;
        private SerializedProperty _propOpacityScale;
        private SerializedProperty _propSHOrder;
        private SerializedProperty _propSHOnly;
        private SerializedProperty _propSortNthFrame;
        private SerializedProperty _propRenderMode;
        private SerializedProperty _propPointDisplaySize;
        private SerializedProperty _propCutouts;
        private SerializedProperty _propShaderSplats;
        private SerializedProperty _propShaderComposite;
        private SerializedProperty _propShaderDebugPoints;
        private SerializedProperty _propShaderDebugBoxes;
        private SerializedProperty _propCsSplatUtilities;

        private bool _resourcesExpanded;
        private int _cameraIndex;

        private bool _exportBakeTransform;

        private static int s_editStatsUpdateCounter;

        private static HashSet<GaussianSplatRendererEditor> s_allEditors = new();

        public static void BumpGUICounter()
        {
            ++s_editStatsUpdateCounter;
        }

        public static void RepaintAll()
        {
            foreach (var e in s_allEditors)
            {
                e.Repaint();
            }
        }

        public void OnEnable()
        {
            _exportBakeTransform = EditorPrefs.GetBool(kPrefExportBake, false);

            _propSplat = serializedObject.FindProperty(nameof(GaussianSplatRenderer.Splat));
            _propRenderOrder = serializedObject.FindProperty(nameof(GaussianSplatRenderer.RenderOrder));
            _propSplatScale = serializedObject.FindProperty(nameof(GaussianSplatRenderer.SplatScale));
            _propOpacityScale = serializedObject.FindProperty(nameof(GaussianSplatRenderer.OpacityScale));
            _propSHOrder = serializedObject.FindProperty(nameof(GaussianSplatRenderer.SHOrder));
            _propSHOnly = serializedObject.FindProperty(nameof(GaussianSplatRenderer.SHOnly));
            _propSortNthFrame = serializedObject.FindProperty(nameof(GaussianSplatRenderer.SortNthFrame));
            _propRenderMode = serializedObject.FindProperty(nameof(GaussianSplatRenderer.Mode));
            _propPointDisplaySize = serializedObject.FindProperty(nameof(GaussianSplatRenderer.PointDisplaySize));
            _propCutouts = serializedObject.FindProperty(nameof(GaussianSplatRenderer.Cutouts));
            _propShaderSplats = serializedObject.FindProperty(nameof(GaussianSplatRenderer.ShaderSplats));
            _propShaderComposite = serializedObject.FindProperty(nameof(GaussianSplatRenderer.ShaderComposite));
            _propShaderDebugPoints = serializedObject.FindProperty(nameof(GaussianSplatRenderer.ShaderDebugPoints));
            _propShaderDebugBoxes = serializedObject.FindProperty(nameof(GaussianSplatRenderer.ShaderDebugBoxes));
            _propCsSplatUtilities = serializedObject.FindProperty(nameof(GaussianSplatRenderer.CsSplatUtilities));

            s_allEditors.Add(this);
        }

        public void OnDisable()
        {
            s_allEditors.Remove(this);
        }

        public override void OnInspectorGUI()
        {
            var gs = target as GaussianSplatRenderer;
            if (!gs)
                return;

            serializedObject.Update();

            GUILayout.Label("Splat Data", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(_propSplat);

            if (!gs.HasValidSplat)
            {
                var msg = gs.Splat != null && gs.Splat.FormatVersion != GaussianSplat.kCurrentVersion
                    ? "Gaussian Splat version is not compatible, please recreate the splat"
                    : "Gaussian Splat is not assigned or is empty";
                EditorGUILayout.HelpBox(msg, MessageType.Error);
            }

            EditorGUILayout.Space();
            GUILayout.Label("Render Options", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(_propRenderOrder);
            EditorGUILayout.PropertyField(_propSplatScale);
            EditorGUILayout.PropertyField(_propOpacityScale);
            EditorGUILayout.PropertyField(_propSHOrder);
            EditorGUILayout.PropertyField(_propSHOnly);
            EditorGUILayout.PropertyField(_propSortNthFrame);

            EditorGUILayout.Space();
            GUILayout.Label("Debugging Tweaks", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(_propRenderMode);
            if (_propRenderMode.intValue is (int)GaussianSplatRenderer.RenderMode.DebugPoints or (int)GaussianSplatRenderer.RenderMode.DebugPointIndices)
                EditorGUILayout.PropertyField(_propPointDisplaySize);

            EditorGUILayout.Space();
            _resourcesExpanded = EditorGUILayout.Foldout(_resourcesExpanded, "Resources", true, EditorStyles.foldoutHeader);
            if (_resourcesExpanded)
            {
                EditorGUILayout.PropertyField(_propShaderSplats);
                EditorGUILayout.PropertyField(_propShaderComposite);
                EditorGUILayout.PropertyField(_propShaderDebugPoints);
                EditorGUILayout.PropertyField(_propShaderDebugBoxes);
                EditorGUILayout.PropertyField(_propCsSplatUtilities);
            }
            bool validAndEnabled = gs && gs.enabled && gs.gameObject.activeInHierarchy && gs.HasValidSplat;
            if (validAndEnabled && !gs.HasValidRenderSetup)
            {
                EditorGUILayout.HelpBox("Shader resources are not set up", MessageType.Error);
                validAndEnabled = false;
            }

            if (validAndEnabled && targets.Length == 1)
            {
                EditCameras(gs);
                EditGUI(gs);
            }
            if (validAndEnabled && targets.Length > 1)
            {
                MultiEditGUI();
            }

            serializedObject.ApplyModifiedProperties();
        }

        private void EditCameras(GaussianSplatRenderer gs)
        {
            var splat = gs.Splat;
            var cameras = splat.Cameras;
            if (cameras != null && cameras.Length != 0)
            {
                EditorGUILayout.Space();
                GUILayout.Label("Cameras", EditorStyles.boldLabel);
                var camIndex = EditorGUILayout.IntSlider("Camera", _cameraIndex, 0, cameras.Length - 1);
                camIndex = math.clamp(camIndex, 0, cameras.Length - 1);
                if (camIndex != _cameraIndex)
                {
                    _cameraIndex = camIndex;
                    gs.ActivateCamera(camIndex);
                }
            }
        }

        private void MultiEditGUI()
        {
            DrawSeparator();
            CountTargetSplats(out var totalSplats, out var totalObjects);
            EditorGUILayout.LabelField("Total Objects", $"{totalObjects}");
            EditorGUILayout.LabelField("Total Splats", $"{totalSplats:N0}");
            if (totalSplats > GaussianSplat.kMaxSplats)
            {
                EditorGUILayout.HelpBox($"Can't merge, too many splats (max. supported {GaussianSplat.kMaxSplats:N0})", MessageType.Warning);
                return;
            }

            var targetGs = (GaussianSplatRenderer) target;
            if (!targetGs || !targetGs.HasValidSplat || !targetGs.isActiveAndEnabled)
            {
                EditorGUILayout.HelpBox($"Can't merge into {target.name} (no splat or disable)", MessageType.Warning);
                return;
            }

            if (targetGs.Splat.ChunkData != null)
            {
                EditorGUILayout.HelpBox($"Can't merge into {target.name} (needs to use Very High quality preset)", MessageType.Warning);
                return;
            }
            if (GUILayout.Button($"Merge into {target.name}"))
            {
                MergeSplatObjects();
            }
        }

        private void CountTargetSplats(out int totalSplats, out int totalObjects)
        {
            totalObjects = 0;
            totalSplats = 0;
            foreach (var obj in targets)
            {
                var gs = obj as GaussianSplatRenderer;
                if (!gs || !gs.HasValidSplat || !gs.isActiveAndEnabled)
                    continue;
                ++totalObjects;
                totalSplats += gs.SplatCount;
            }
        }

        private void MergeSplatObjects()
        {
            CountTargetSplats(out var totalSplats, out _);
            if (totalSplats > GaussianSplat.kMaxSplats)
                return;
            var targetGs = (GaussianSplatRenderer) target;

            int copyDstOffset = targetGs.SplatCount;
            targetGs.EditSetSplatCount(totalSplats);
            foreach (var obj in targets)
            {
                var gs = obj as GaussianSplatRenderer;
                if (!gs || !gs.HasValidSplat || !gs.isActiveAndEnabled)
                    continue;
                if (gs == targetGs)
                    continue;
                gs.EditCopySplatsInto(targetGs, 0, copyDstOffset, gs.SplatCount);
                copyDstOffset += gs.SplatCount;
                gs.gameObject.SetActive(false);
            }
            Debug.Assert(copyDstOffset == totalSplats, $"Merge count mismatch, {copyDstOffset} vs {totalSplats}");
            Selection.activeObject = targetGs;
        }

        private void EditGUI(GaussianSplatRenderer gs)
        {
            ++s_editStatsUpdateCounter;

            DrawSeparator();
            bool wasToolActive = ToolManager.activeContextType == typeof(GaussianToolContext);
            GUILayout.BeginHorizontal();
            bool isToolActive = GUILayout.Toggle(wasToolActive, "Edit", EditorStyles.miniButton);
            using (new EditorGUI.DisabledScope(!gs.EditModified))
            {
                if (GUILayout.Button("Reset", GUILayout.ExpandWidth(false)))
                {
                    if (EditorUtility.DisplayDialog("Reset Splat Modifications?",
                            $"This will reset edits of {gs.name} to match the {gs.Splat.name} splat. Continue?",
                            "Yes, reset", "Cancel"))
                    {
                        gs.enabled = false;
                        gs.enabled = true;
                    }
                }
            }

            GUILayout.EndHorizontal();
            if (!wasToolActive && isToolActive)
            {
                ToolManager.SetActiveContext<GaussianToolContext>();
                if (Tools.current == Tool.View)
                    Tools.current = Tool.Move;
            }

            if (wasToolActive && !isToolActive)
            {
                ToolManager.SetActiveContext<GameObjectToolContext>();
            }

            if (isToolActive && gs.Splat.ChunkData != null)
            {
                EditorGUILayout.HelpBox("Splat move/rotate/scale tools need Very High splat quality preset", MessageType.Warning);
            }

            EditorGUILayout.Space();
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Add Cutout"))
            {
                GaussianCutout cutout = ObjectFactory.CreateGameObject("GSCutout", typeof(GaussianCutout)).GetComponent<GaussianCutout>();
                Transform cutoutTr = cutout.transform;
                cutoutTr.SetParent(gs.transform, false);
                cutoutTr.localScale = (gs.Splat.BoundsMax - gs.Splat.BoundsMin) * 0.25f;
                gs.Cutouts ??= Array.Empty<GaussianCutout>();
                ArrayUtility.Add(ref gs.Cutouts, cutout);
                gs.UpdateEditCountsAndBounds();
                EditorUtility.SetDirty(gs);
                Selection.activeGameObject = cutout.gameObject;
            }
            if (GUILayout.Button("Use All Cutouts"))
            {
                gs.Cutouts = FindObjectsByType<GaussianCutout>(FindObjectsSortMode.InstanceID);
                gs.UpdateEditCountsAndBounds();
                EditorUtility.SetDirty(gs);
            }

            if (GUILayout.Button("No Cutouts"))
            {
                gs.Cutouts = Array.Empty<GaussianCutout>();
                gs.UpdateEditCountsAndBounds();
                EditorUtility.SetDirty(gs);
            }
            GUILayout.EndHorizontal();
            EditorGUILayout.PropertyField(_propCutouts);

            bool hasCutouts = gs.Cutouts != null && gs.Cutouts.Length != 0;
            bool modifiedOrHasCutouts = gs.EditModified || hasCutouts;

            var splat = gs.Splat;
            EditorGUILayout.Space();
            EditorGUI.BeginChangeCheck();
            _exportBakeTransform = EditorGUILayout.Toggle("Export in world space", _exportBakeTransform);
            if (EditorGUI.EndChangeCheck())
            {
                EditorPrefs.SetBool(kPrefExportBake, _exportBakeTransform);
            }

            if (GUILayout.Button("Export PLY"))
                ExportPlyFile(gs, _exportBakeTransform);
            if (splat.PosFormat > GaussianSplat.VectorFormat.Norm16 ||
                splat.ScaleFormat > GaussianSplat.VectorFormat.Norm16 ||
                splat.ColFormat > GaussianSplat.ColorFormat.Float16X4 ||
                splat.ShFormat > GaussianSplat.SHFormat.Float16)
            {
                EditorGUILayout.HelpBox(
                    "It is recommended to use High or VeryHigh quality preset for editing splats, lower levels are lossy",
                    MessageType.Warning);
            }

            bool displayEditStats = isToolActive || modifiedOrHasCutouts;
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Splats", $"{gs.SplatCount:N0}");
            if (displayEditStats)
            {
                EditorGUILayout.LabelField("Cut", $"{gs.EditCutSplats:N0}");
                EditorGUILayout.LabelField("Deleted", $"{gs.EditDeletedSplats:N0}");
                EditorGUILayout.LabelField("Selected", $"{gs.EditSelectedSplats:N0}");
                if (hasCutouts)
                {
                    if (s_editStatsUpdateCounter > 10)
                    {
                        gs.UpdateEditCountsAndBounds();
                        s_editStatsUpdateCounter = 0;
                    }
                }
            }
        }

        private static void DrawSeparator()
        {
            EditorGUILayout.Space(12f, true);
            GUILayout.Box(GUIContent.none, "sv_iconselector_sep", GUILayout.Height(2), GUILayout.ExpandWidth(true));
            EditorGUILayout.Space();
        }

        private bool HasFrameBounds()
        {
            return true;
        }

        private Bounds OnGetFrameBounds()
        {
            var gs = target as GaussianSplatRenderer;
            if (!gs || !gs.HasValidRenderSetup)
                return new Bounds(Vector3.zero, Vector3.one);
            Bounds bounds = default;
            bounds.SetMinMax(gs.Splat.BoundsMin, gs.Splat.BoundsMax);
            if (gs.EditSelectedSplats > 0)
            {
                bounds = gs.EditSelectedBounds;
            }
            bounds.extents *= 0.7f;
            return TransformBounds(gs.transform, bounds);
        }

        public static Bounds TransformBounds(Transform tr, Bounds bounds )
        {
            var center = tr.TransformPoint(bounds.center);

            var ext = bounds.extents;
            var axisX = tr.TransformVector(ext.x, 0, 0);
            var axisY = tr.TransformVector(0, ext.y, 0);
            var axisZ = tr.TransformVector(0, 0, ext.z);

            // sum their absolute value to get the world extents
            ext.x = Mathf.Abs(axisX.x) + Mathf.Abs(axisY.x) + Mathf.Abs(axisZ.x);
            ext.y = Mathf.Abs(axisX.y) + Mathf.Abs(axisY.y) + Mathf.Abs(axisZ.y);
            ext.z = Mathf.Abs(axisX.z) + Mathf.Abs(axisY.z) + Mathf.Abs(axisZ.z);

            return new Bounds { center = center, extents = ext };
        }

        private static unsafe void ExportPlyFile(GaussianSplatRenderer gs, bool bakeTransform)
        {
            var path = EditorUtility.SaveFilePanel(
                "Export Gaussian Splat PLY file", "", $"{gs.Splat.name}-edit.ply", "ply");
            if (string.IsNullOrWhiteSpace(path))
                return;

            int kSplatSize = UnsafeUtility.SizeOf<Utils.InputSplatData>();
            using var gpuData = new GraphicsBuffer(GraphicsBuffer.Target.Structured, gs.SplatCount, kSplatSize);

            if (!gs.EditExportData(gpuData, bakeTransform))
                return;

            Utils.InputSplatData[] data = new Utils.InputSplatData[gpuData.count];
            gpuData.GetData(data);

            var gpuDeleted = gs.GpuEditDeleted;
            uint[] deleted = new uint[gpuDeleted.count];
            gpuDeleted.GetData(deleted);

            // count non-deleted splats
            int aliveCount = 0;
            for (int i = 0; i < data.Length; ++i)
            {
                int wordIdx = i >> 5;
                int bitIdx = i & 31;
                bool isDeleted = (deleted[wordIdx] & (1u << bitIdx)) != 0;
                bool isCutout = data[i].Nor.sqrMagnitude > 0;
                if (!isDeleted && !isCutout)
                    ++aliveCount;
            }

            using FileStream fs = new FileStream(path, FileMode.Create, FileAccess.Write);
            // note: this is a long string! but we don't use multiline literal because we want guaranteed LF line ending
            var header = $"ply\nformat binary_little_endian 1.0\nelement vertex {aliveCount}\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\nproperty float f_rest_0\nproperty float f_rest_1\nproperty float f_rest_2\nproperty float f_rest_3\nproperty float f_rest_4\nproperty float f_rest_5\nproperty float f_rest_6\nproperty float f_rest_7\nproperty float f_rest_8\nproperty float f_rest_9\nproperty float f_rest_10\nproperty float f_rest_11\nproperty float f_rest_12\nproperty float f_rest_13\nproperty float f_rest_14\nproperty float f_rest_15\nproperty float f_rest_16\nproperty float f_rest_17\nproperty float f_rest_18\nproperty float f_rest_19\nproperty float f_rest_20\nproperty float f_rest_21\nproperty float f_rest_22\nproperty float f_rest_23\nproperty float f_rest_24\nproperty float f_rest_25\nproperty float f_rest_26\nproperty float f_rest_27\nproperty float f_rest_28\nproperty float f_rest_29\nproperty float f_rest_30\nproperty float f_rest_31\nproperty float f_rest_32\nproperty float f_rest_33\nproperty float f_rest_34\nproperty float f_rest_35\nproperty float f_rest_36\nproperty float f_rest_37\nproperty float f_rest_38\nproperty float f_rest_39\nproperty float f_rest_40\nproperty float f_rest_41\nproperty float f_rest_42\nproperty float f_rest_43\nproperty float f_rest_44\nproperty float opacity\nproperty float scale_0\nproperty float scale_1\nproperty float scale_2\nproperty float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\nend_header\n";
            fs.Write(Encoding.UTF8.GetBytes(header));
            for (int i = 0; i < data.Length; ++i)
            {
                int wordIdx = i >> 5;
                int bitIdx = i & 31;
                bool isDeleted = (deleted[wordIdx] & (1u << bitIdx)) != 0;
                bool isCutout = data[i].Nor.sqrMagnitude > 0;
                if (!isDeleted && !isCutout)
                {
                    var splat = data[i];
                    byte* ptr = (byte*)&splat;
                    fs.Write(new ReadOnlySpan<byte>(ptr, kSplatSize));
                }
            }

            Debug.Log($"Exported PLY {path} with {aliveCount:N0} splats");
        }
    }
}