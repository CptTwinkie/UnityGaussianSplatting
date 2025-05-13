// SPDX-License-Identifier: MIT

using GaussianSplatting.Runtime;
using Unity.Collections.LowLevel.Unsafe;
using UnityEditor;
using UnityEngine;

namespace GaussianSplatting.Editor
{
    [CustomEditor(typeof(GaussianSplat))]
    [CanEditMultipleObjects]
    public class GaussianSplatEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            var gs = target as GaussianSplat;
            if (!gs)
                return;

            using var _ = new EditorGUI.DisabledScope(true);

            if (targets.Length == 1)
            {
                SingleSplatGUI(gs);
            }
            else
            {
                int totalCount = 0;
                foreach (var tgt in targets)
                {
                    var gss = tgt as GaussianSplat;
                    if (gss)
                    {
                        totalCount += gss.SplatCount;
                    }
                }
                EditorGUILayout.TextField("Total Splats", $"{totalCount:N0}");
            }
        }

        private static void SingleSplatGUI(GaussianSplat gs)
        {
            var splatCount = gs.SplatCount;
            EditorGUILayout.TextField("Splats", $"{splatCount:N0}");
            var prevBackColor = GUI.backgroundColor;
            if (gs.FormatVersion != GaussianSplat.kCurrentVersion)
            {
                GUI.backgroundColor *= Color.red;
            }
            EditorGUILayout.IntField("Version", gs.FormatVersion);
            GUI.backgroundColor = prevBackColor;

            long sizePos = gs.PosData != null ? gs.PosData.dataSize : 0;
            long sizeOther = gs.OtherData != null ? gs.OtherData.dataSize : 0;
            long sizeCol = gs.ColorData != null ? gs.ColorData.dataSize : 0;
            long sizeSH = GaussianSplat.CalcSHDataSize(gs.SplatCount, gs.ShFormat);
            long sizeChunk = gs.ChunkData != null ? gs.ChunkData.dataSize : 0;

            EditorGUILayout.TextField("Memory", EditorUtility.FormatBytes(sizePos + sizeOther + sizeSH + sizeCol + sizeChunk));
            EditorGUI.indentLevel++;
            EditorGUILayout.TextField("Positions", $"{EditorUtility.FormatBytes(sizePos)}  ({gs.PosFormat})");
            EditorGUILayout.TextField("Other", $"{EditorUtility.FormatBytes(sizeOther)}  ({gs.ScaleFormat})");
            EditorGUILayout.TextField("Base color", $"{EditorUtility.FormatBytes(sizeCol)}  ({gs.ColFormat})");
            EditorGUILayout.TextField("SHs", $"{EditorUtility.FormatBytes(sizeSH)}  ({gs.ShFormat})");
            EditorGUILayout.TextField("Chunks", $"{EditorUtility.FormatBytes(sizeChunk)}  ({UnsafeUtility.SizeOf<GaussianSplat.ChunkInfo>()} B/chunk)");
            EditorGUI.indentLevel--;

            EditorGUILayout.Vector3Field("Bounds Min", gs.BoundsMin);
            EditorGUILayout.Vector3Field("Bounds Max", gs.BoundsMax);

            EditorGUILayout.TextField("Data Hash", gs.DataHash.ToString());
        }
    }
}
