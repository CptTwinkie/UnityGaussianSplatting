// SPDX-License-Identifier: MIT

using GaussianSplatting.Runtime;
using UnityEditor.EditorTools;
using UnityEngine;

namespace GaussianSplatting.Editor
{
    internal abstract class GaussianTool : EditorTool
    {
        protected GaussianSplatRenderer GetRenderer()
        {
            var gs = target as GaussianSplatRenderer;
            return !gs || !gs.HasValidSplat || !gs.HasValidRenderSetup ? null : gs;
        }

        protected bool CanBeEdited()
        {
            var gs = GetRenderer();
            if (!gs)
                return false;
            return gs.Splat.ChunkData == null; // need to be lossless / non-chunked for editing
        }

        protected bool HasSelection()
        {
            var gs = GetRenderer();
            if (!gs)
                return false;
            return gs.EditSelectedSplats > 0;
        }

        protected Vector3 GetSelectionCenterLocal()
        {
            var gs = GetRenderer();
            if (!gs || gs.EditSelectedSplats == 0)
                return Vector3.zero;
            return gs.EditSelectedBounds.center;
        }
    }
}
