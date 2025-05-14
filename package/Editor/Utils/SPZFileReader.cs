// SPDX-License-Identifier: MIT

using System.IO;
using Unity.Collections;
using System.IO.Compression;
using GaussianSplatting.Runtime;
using Unity.Burst;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace GaussianSplatting.Editor.Utils
{
    // reads Niantic/Scaniverse .SPZ files:
    // https://github.com/nianticlabs/spz
    // https://scaniverse.com/spz
    [BurstCompile]
    public static class SpzFileReader
    {
        private struct SpzHeader {
            public uint Magic; // 0x5053474e "NGSP"
            public uint Version; // 2
            public uint NumPoints;
            public uint SHFracbitsFlagsReserved;
        };
        public static void ReadFileHeader(string filePath, out int vertexCount)
        {
            vertexCount = 0;
            if (!File.Exists(filePath))
                return;
            using var fs = File.OpenRead(filePath);
            using var gz = new GZipStream(fs, CompressionMode.Decompress);
            ReadHeaderImpl(filePath, gz, out vertexCount, out _, out _, out _);
        }

        private static void ReadHeaderImpl(string filePath, Stream fs, out int vertexCount, out int shLevel, out int fractBits, out int flags)
        {
            var header = new NativeArray<SpzHeader>(1, Allocator.Temp);
            var readBytes = fs.Read(header.Reinterpret<byte>(16));
            if (readBytes != 16)
                throw new IOException($"SPZ {filePath} read error, failed to read header");

            if (header[0].Magic != 0x5053474e)
                throw new IOException($"SPZ {filePath} read error, header magic unexpected {header[0].Magic}");
            if (header[0].Version != 2)
                throw new IOException($"SPZ {filePath} read error, header version unexpected {header[0].Version}");

            vertexCount = (int)header[0].NumPoints;
            shLevel = (int)(header[0].SHFracbitsFlagsReserved & 0xFF);
            fractBits = (int)((header[0].SHFracbitsFlagsReserved >> 8) & 0xFF);
            flags = (int)((header[0].SHFracbitsFlagsReserved >> 16) & 0xFF);
        }

        private static int SHCoeffsForLevel(int level)
        {
            return level switch
            {
                0 => 0,
                1 => 3,
                2 => 8,
                3 => 15,
                _ => 0
            };
        }

        public static void ReadFile(string filePath, out NativeArray<InputSplatData> splats)
        {
            using var fs = File.OpenRead(filePath);
            using var gz = new GZipStream(fs, CompressionMode.Decompress);
            ReadHeaderImpl(filePath, gz, out var splatCount, out var shLevel, out var fractBits, out var flags);

            if (splatCount < 1 || splatCount > 10_000_000) // 10M hardcoded in SPZ code
                throw new IOException($"SPZ {filePath} read error, out of range splat count {splatCount}");
            if (shLevel < 0 || shLevel > 3)
                throw new IOException($"SPZ {filePath} read error, out of range SH level {shLevel}");
            if (fractBits < 0 || fractBits > 24)
                throw new IOException($"SPZ {filePath} read error, out of range fractional bits {fractBits}");

            // allocate temporary storage
            int shCoeffs = SHCoeffsForLevel(shLevel);
            NativeArray<byte> packedPos = new(splatCount * 3 * 3, Allocator.Persistent);
            NativeArray<byte> packedScale = new(splatCount * 3, Allocator.Persistent);
            NativeArray<byte> packedRot = new(splatCount * 3, Allocator.Persistent);
            NativeArray<byte> packedAlpha = new(splatCount, Allocator.Persistent);
            NativeArray<byte> packedCol = new(splatCount * 3, Allocator.Persistent);
            NativeArray<byte> packedSh = new(splatCount * 3 * shCoeffs, Allocator.Persistent);

            // read file contents into temporaries
            bool readOk = true;
            readOk &= gz.Read(packedPos) == packedPos.Length;
            readOk &= gz.Read(packedAlpha) == packedAlpha.Length;
            readOk &= gz.Read(packedCol) == packedCol.Length;
            readOk &= gz.Read(packedScale) == packedScale.Length;
            readOk &= gz.Read(packedRot) == packedRot.Length;
            readOk &= gz.Read(packedSh) == packedSh.Length;

            // unpack into full splat data
            splats = new NativeArray<InputSplatData>(splatCount, Allocator.Persistent);
            UnpackDataJob job = new UnpackDataJob();
            job.PackedPos = packedPos;
            job.PackedScale = packedScale;
            job.PackedRot = packedRot;
            job.PackedAlpha = packedAlpha;
            job.PackedCol = packedCol;
            job.PackedSh = packedSh;
            job.SHCoeffs = shCoeffs;
            job.FractScale = 1.0f / (1 << fractBits);
            job.Splats = splats;
            job.Schedule(splatCount, 4096).Complete();

            // cleanup
            packedPos.Dispose();
            packedScale.Dispose();
            packedRot.Dispose();
            packedAlpha.Dispose();
            packedCol.Dispose();
            packedSh.Dispose();

            if (!readOk)
            {
                splats.Dispose();
                throw new IOException($"SPZ {filePath} read error, file smaller than it should be");
            }
        }

        [BurstCompile]
        private struct UnpackDataJob : IJobParallelFor
        {
            [NativeDisableParallelForRestriction] [ReadOnly] public NativeArray<byte> PackedPos;
            [NativeDisableParallelForRestriction] [ReadOnly] public NativeArray<byte> PackedScale;
            [NativeDisableParallelForRestriction] [ReadOnly] public NativeArray<byte> PackedRot;
            [NativeDisableParallelForRestriction] [ReadOnly] public NativeArray<byte> PackedAlpha;
            [NativeDisableParallelForRestriction] [ReadOnly] public NativeArray<byte> PackedCol;
            [NativeDisableParallelForRestriction] [ReadOnly] public NativeArray<byte> PackedSh;
            public float FractScale;
            public int SHCoeffs;
            public NativeArray<InputSplatData> Splats;

            public void Execute(int index)
            {
                var splat = Splats[index];

                splat.Pos = new Vector3(UnpackFloat(index * 3 + 0) * FractScale, UnpackFloat(index * 3 + 1) * FractScale, UnpackFloat(index * 3 + 2) * FractScale);

                splat.Scale = new Vector3(PackedScale[index * 3 + 0], PackedScale[index * 3 + 1], PackedScale[index * 3 + 2]) / 16.0f - new Vector3(10.0f, 10.0f, 10.0f);
                splat.Scale = GaussianUtils.LinearScale(splat.Scale);

                Vector3 xyz = new Vector3(PackedRot[index * 3 + 0], PackedRot[index * 3 + 1], PackedRot[index * 3 + 2]) * (1.0f / 127.5f) - new Vector3(1, 1, 1);
                float w = math.sqrt(math.max(0.0f, 1.0f - xyz.sqrMagnitude));
                var q = new float4(xyz.x, xyz.y, xyz.z, w);
                var qq = math.normalize(q);
                qq = GaussianUtils.PackSmallest3Rotation(qq);
                splat.Rot = new Quaternion(qq.x, qq.y, qq.z, qq.w);

                splat.Opacity = PackedAlpha[index] / 255.0f;

                Vector3 col = new Vector3(PackedCol[index * 3 + 0], PackedCol[index * 3 + 1], PackedCol[index * 3 + 2]);
                col = col / 255.0f - new Vector3(0.5f, 0.5f, 0.5f);
                col /= 0.15f;
                splat.Dc0 = GaussianUtils.SH0ToColor(col);

                int shIdx = index * SHCoeffs * 3;
                splat.SH1 = UnpackSH(shIdx); shIdx += 3;
                splat.SH2 = UnpackSH(shIdx); shIdx += 3;
                splat.SH3 = UnpackSH(shIdx); shIdx += 3;
                splat.SH4 = UnpackSH(shIdx); shIdx += 3;
                splat.SH5 = UnpackSH(shIdx); shIdx += 3;
                splat.SH6 = UnpackSH(shIdx); shIdx += 3;
                splat.SH7 = UnpackSH(shIdx); shIdx += 3;
                splat.SH8 = UnpackSH(shIdx); shIdx += 3;
                splat.SH9 = UnpackSH(shIdx); shIdx += 3;
                splat.SHA = UnpackSH(shIdx); shIdx += 3;
                splat.SHB = UnpackSH(shIdx); shIdx += 3;
                splat.SHC = UnpackSH(shIdx); shIdx += 3;
                splat.SHD = UnpackSH(shIdx); shIdx += 3;
                splat.SHE = UnpackSH(shIdx); shIdx += 3;
                splat.SHF = UnpackSH(shIdx); shIdx += 3;

                Splats[index] = splat;
            }

            private float UnpackFloat(int idx)
            {
                int fx = PackedPos[idx * 3 + 0] | (PackedPos[idx * 3 + 1] << 8) | (PackedPos[idx * 3 + 2] << 16);
                fx |= (fx & 0x800000) != 0 ? -16777216 : 0; // sign extension with 0xff000000
                return fx;
            }

            private Vector3 UnpackSH(int idx)
            {
                Vector3 sh = new Vector3(PackedSh[idx], PackedSh[idx + 1], PackedSh[idx + 2]) - new Vector3(128.0f, 128.0f, 128.0f);
                sh /= 128.0f;
                return sh;
            }
        }

    }
}
