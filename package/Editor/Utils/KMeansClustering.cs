// SPDX-License-Identifier: MIT

using System;
using Unity.Burst;
using Unity.Burst.Intrinsics;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Profiling;
using Unity.Profiling.LowLevel;

namespace GaussianSplatting.Editor.Utils
{
    // Implementation of "Mini Batch" k-means clustering ("Web-Scale K-Means Clustering", Sculley 2010)
    // using k-means++ for cluster initialization.
    [BurstCompile]
    public struct KMeansClustering
    {
        private static ProfilerMarker s_profCalculate             = new(ProfilerCategory.Render, "KMeans.Calculate", MarkerFlags.SampleGPU);
        private static ProfilerMarker s_profPlusPlus              = new(ProfilerCategory.Render, "KMeans.InitialPlusPlus", MarkerFlags.SampleGPU);
        private static ProfilerMarker s_profInitialDistanceSum    = new(ProfilerCategory.Render, "KMeans.Initialize.DistanceSum", MarkerFlags.SampleGPU);
        private static ProfilerMarker s_profInitialPickPoint      = new(ProfilerCategory.Render, "KMeans.Initialize.PickPoint", MarkerFlags.SampleGPU);
        private static ProfilerMarker s_profInitialDistanceUpdate = new(ProfilerCategory.Render, "KMeans.Initialize.DistanceUpdate", MarkerFlags.SampleGPU);
        private static ProfilerMarker s_profAssignClusters        = new(ProfilerCategory.Render, "KMeans.AssignClusters", MarkerFlags.SampleGPU);
        private static ProfilerMarker s_profUpdateMeans           = new(ProfilerCategory.Render, "KMeans.UpdateMeans", MarkerFlags.SampleGPU);

        public static bool Calculate(int dim, NativeArray<float> inputData, int batchSize, float passesOverData, Func<float,bool> progress, NativeArray<float> outClusterMeans, NativeArray<int> outDataLabels)
        {
            // Parameter checks
            if (dim < 1) throw new InvalidOperationException($"KMeans: dimensionality has to be >= 1, was {dim}");
            if (batchSize < 1) throw new InvalidOperationException($"KMeans: batch size has to be >= 1, was {batchSize}");
            if (passesOverData < 0.0001f) throw new InvalidOperationException($"KMeans: passes over data must be positive, was {passesOverData}");
            if (inputData.Length % dim != 0) throw new InvalidOperationException($"KMeans: input length must be multiple of dim={dim}, was {inputData.Length}");
            if (outClusterMeans.Length % dim != 0) throw new InvalidOperationException($"KMeans: output means length must be multiple of dim={dim}, was {outClusterMeans.Length}");
            int dataSize = inputData.Length / dim;
            int k = outClusterMeans.Length / dim;
            if (k < 1) throw new InvalidOperationException($"KMeans: cluster count length must be at least 1, was {k}");
            if (dataSize < k) throw new InvalidOperationException($"KMeans: input length ({inputData.Length}) must at least as long as clusters ({outClusterMeans.Length})");
            if (dataSize != outDataLabels.Length) throw new InvalidOperationException($"KMeans: output labels length must be {dataSize}, was {outDataLabels.Length}");

            using var prof = s_profCalculate.Auto();
            batchSize = math.min(dataSize, batchSize);
            uint rngState = 1;

            // Do initial cluster placement
            int initBatchSize = 10 * k;
            const int kinitAttempts = 3;
            if (!InitializeCentroids(dim, inputData, initBatchSize, ref rngState, kinitAttempts, outClusterMeans, progress))
                return false;

            NativeArray<float> counts      = new(k, Allocator.TempJob);
            NativeArray<float> batchPoints = new(batchSize * dim, Allocator.TempJob);
            NativeArray<int> batchClusters = new(batchSize, Allocator.TempJob);

            bool cancelled = false;
            for (float calcDone = 0.0f, calcLimit = dataSize * passesOverData; calcDone < calcLimit; calcDone += batchSize)
            {
                if (progress != null && !progress(0.3f + calcDone / calcLimit * 0.4f))
                {
                    cancelled = true;
                    break;
                }

                // generate a batch of random input points
                MakeRandomBatch(dim, inputData, ref rngState, batchPoints);

                // find which of the current centroids each batch point is closest to
                {
                    using var profPart = s_profAssignClusters.Auto();
                    AssignClustersJob job = new AssignClustersJob
                    {
                        Dim = dim,
                        Data = batchPoints,
                        Means = outClusterMeans,
                        IndexOffset = 0,
                        Clusters = batchClusters,
                    };
                    job.Schedule(batchSize, 1).Complete();
                }

                // update the centroids
                {
                    using var profPart = s_profUpdateMeans.Auto();
                    UpdateCentroidsJob job = new UpdateCentroidsJob
                    {
                        Clusters = outClusterMeans,
                        Dim = dim,
                        Counts = counts,
                        BatchSize = batchSize,
                        BatchClusters = batchClusters,
                        BatchPoints = batchPoints
                    };
                    job.Schedule().Complete();
                }
            }

            // finally find out closest clusters for all input points
            {
                using var profPart = s_profAssignClusters.Auto();
                const int kassignBatchCount = 256 * 1024;
                AssignClustersJob job = new AssignClustersJob
                {
                    Dim = dim,
                    Data = inputData,
                    Means = outClusterMeans,
                    IndexOffset = 0,
                    Clusters = outDataLabels,
                };
                for (int i = 0; i < dataSize; i += kassignBatchCount)
                {
                    if (progress != null && !progress(0.7f + (float) i / dataSize * 0.3f))
                    {
                        cancelled = true;
                        break;
                    }
                    job.IndexOffset = i;
                    job.Schedule(math.min(kassignBatchCount, dataSize - i), 512).Complete();
                }
            }

            counts.Dispose();
            batchPoints.Dispose();
            batchClusters.Dispose();
            return !cancelled;
        }

        private static unsafe float DistanceSquared(int dim, NativeArray<float> a, int aIndex, NativeArray<float> b, int bIndex)
        {
            aIndex *= dim;
            bIndex *= dim;
            float d = 0;
            if (X86.Avx.IsAvxSupported)
            {
                // 8x wide with AVX
                int i = 0;
                float* aptr = (float*) a.GetUnsafeReadOnlyPtr() + aIndex;
                float* bptr = (float*) b.GetUnsafeReadOnlyPtr() + bIndex;
                for (; i + 7 < dim; i += 8)
                {
                    v256 va = X86.Avx.mm256_loadu_ps(aptr);
                    v256 vb = X86.Avx.mm256_loadu_ps(bptr);
                    v256 vd = X86.Avx.mm256_sub_ps(va, vb);
                    vd = X86.Avx.mm256_mul_ps(vd, vd);

                    vd = X86.Avx.mm256_hadd_ps(vd, vd);
                    d += vd.Float0 + vd.Float1 + vd.Float4 + vd.Float5;

                    aptr += 8;
                    bptr += 8;
                }
                // remainder
                for (; i < dim; ++i)
                {
                    float delta = *aptr - *bptr;
                    d += delta * delta;
                    aptr++;
                    bptr++;
                }
            }
            else if (Arm.Neon.IsNeonSupported)
            {
                // 4x wide with NEON
                int i = 0;
                float* aptr = (float*) a.GetUnsafeReadOnlyPtr() + aIndex;
                float* bptr = (float*) b.GetUnsafeReadOnlyPtr() + bIndex;
                for (; i + 3 < dim; i += 4)
                {
                    v128 va = Arm.Neon.vld1q_f32(aptr);
                    v128 vb = Arm.Neon.vld1q_f32(bptr);
                    v128 vd = Arm.Neon.vsubq_f32(va, vb);
                    vd = Arm.Neon.vmulq_f32(vd, vd);

                    d += Arm.Neon.vaddvq_f32(vd);

                    aptr += 4;
                    bptr += 4;
                }
                // remainder
                for (; i < dim; ++i)
                {
                    float delta = *aptr - *bptr;
                    d += delta * delta;
                    aptr++;
                    bptr++;
                }

            }
            else
            {
                for (var i = 0; i < dim; ++i)
                {
                    float delta = a[aIndex + i] - b[bIndex + i];
                    d += delta * delta;
                }
            }

            return d;
        }

        private static unsafe void CopyElem(int dim, NativeArray<float> src, int srcIndex, NativeArray<float> dst, int dstIndex)
        {
            UnsafeUtility.MemCpy((float*) dst.GetUnsafePtr() + dstIndex * dim,
                (float*) src.GetUnsafeReadOnlyPtr() + srcIndex * dim, dim * 4);
        }

        [BurstCompile]
        private struct ClosestDistanceInitialJob : IJobParallelFor
        {
            public int Dim;
            [ReadOnly] public NativeArray<float> Data;
            [ReadOnly] public NativeArray<float> Means;
            public NativeArray<float> MinDistSq;
            public int PointIndex;
            public void Execute(int index)
            {
                if (index == PointIndex)
                    return;
                MinDistSq[index] = DistanceSquared(Dim, Data, index, Means, 0);
            }
        }

        [BurstCompile]
        private struct ClosestDistanceUpdateJob : IJobParallelFor
        {
            public int Dim;
            [ReadOnly] public NativeArray<float> Data;
            [ReadOnly] public NativeArray<float> Means;
            [ReadOnly] public NativeBitArray Taken;
            public NativeArray<float> MinDistSq;
            public int MeanIndex;
            public void Execute(int index)
            {
                if (Taken.IsSet(index))
                    return;
                float distSq = DistanceSquared(Dim, Data, index, Means, MeanIndex);
                MinDistSq[index] = math.min(MinDistSq[index], distSq);
            }
        }

        [BurstCompile]
        private struct CalcDistSqJob : IJobParallelFor
        {
            public const int kBatchSize = 1024;
            public int DataSize;
            [ReadOnly] public NativeBitArray Taken;
            [ReadOnly] public NativeArray<float> MinDistSq;
            public NativeArray<float> PartialSums;

            public void Execute(int batchIndex)
            {
                int iStart = math.min(batchIndex * kBatchSize, DataSize);
                int iEnd = math.min((batchIndex + 1) * kBatchSize, DataSize);
                float sum = 0;
                for (int i = iStart; i < iEnd; ++i)
                {
                    if (Taken.IsSet(i))
                        continue;
                    sum += MinDistSq[i];
                }

                PartialSums[batchIndex] = sum;
            }
        }

        [BurstCompile]
        private static int PickPointIndex(int dataSize, ref NativeArray<float> partialSums, ref NativeBitArray taken, ref NativeArray<float> minDistSq, float rval)
        {
            // Skip batches until we hit the ones that might have value to pick from: binary search for the batch
            int indexL = 0;
            int indexR = partialSums.Length;
            while (indexL < indexR)
            {
                int indexM = (indexL + indexR) / 2;
                if (partialSums[indexM] < rval)
                    indexL = indexM + 1;
                else
                    indexR = indexM;
            }
            float acc = 0.0f;
            if (indexL > 0)
            {
                acc = partialSums[indexL-1];
            }

            // Now search for the needed point
            int pointIndex = -1;
            for (int i = indexL * CalcDistSqJob.kBatchSize; i < dataSize; ++i)
            {
                if (taken.IsSet(i))
                    continue;
                acc += minDistSq[i];
                if (acc >= rval)
                {
                    pointIndex = i;
                    break;
                }
            }

            // If we have not found a point, pick the last available one
            if (pointIndex < 0)
            {
                for (int i = dataSize - 1; i >= 0; --i)
                {
                    if (taken.IsSet(i))
                        continue;
                    pointIndex = i;
                    break;
                }
            }

            if (pointIndex < 0)
                pointIndex = 0;

            return pointIndex;
        }

        private static void KMeansPlusPlus(int dim, int k, NativeArray<float> data, NativeArray<float> means, NativeArray<float> minDistSq, ref uint rngState)
        {
            using var prof = s_profPlusPlus.Auto();

            int dataSize = data.Length / dim;

            NativeBitArray taken = new NativeBitArray(dataSize, Allocator.TempJob);

            // Select first mean randomly
            int pointIndex = (int)(pcg_random(ref rngState) % dataSize);
            taken.Set(pointIndex, true);
            CopyElem(dim, data, pointIndex, means, 0);

            // For each point: closest squared distance to the picked point
            {
                ClosestDistanceInitialJob job = new ClosestDistanceInitialJob
                {
                    Dim = dim,
                    Data = data,
                    Means = means,
                    MinDistSq = minDistSq,
                    PointIndex = pointIndex
                };
                job.Schedule(dataSize, 1024).Complete();
            }

            int sumBatches = (dataSize + CalcDistSqJob.kBatchSize - 1) / CalcDistSqJob.kBatchSize;
            NativeArray<float> partialSums = new(sumBatches, Allocator.TempJob);
            int resultCount = 1;
            while (resultCount < k)
            {
                // Find total sum of distances of not yet taken points
                float distSqTotal = 0;
                {
                    using var profPart = s_profInitialDistanceSum.Auto();
                    CalcDistSqJob job = new CalcDistSqJob
                    {
                        DataSize = dataSize,
                        Taken = taken,
                        MinDistSq = minDistSq,
                        PartialSums = partialSums
                    };
                    job.Schedule(sumBatches, 1).Complete();
                    for (int i = 0; i < sumBatches; ++i)
                    {
                        distSqTotal += partialSums[i];
                        partialSums[i] = distSqTotal;
                    }
                }

                // Pick a non-taken point, with a probability proportional
                // to distance: points furthest from any cluster are picked more.
                {
                    using var profPart = s_profInitialPickPoint.Auto();
                    float rval = pcg_hash_float(rngState + (uint)resultCount, distSqTotal);
                    pointIndex = PickPointIndex(dataSize, ref partialSums, ref taken, ref minDistSq, rval);
                }

                // Take this point as a new cluster mean
                taken.Set(pointIndex, true);
                CopyElem(dim, data, pointIndex, means, resultCount);
                ++resultCount;

                if (resultCount < k)
                {
                    // Update distances of the points: since it tracks closest one,
                    // calculate distance to the new cluster and update if smaller.
                    using var profPart = s_profInitialDistanceUpdate.Auto();
                    ClosestDistanceUpdateJob job = new ClosestDistanceUpdateJob
                    {
                        Dim = dim,
                        Data = data,
                        Means = means,
                        MinDistSq = minDistSq,
                        Taken = taken,
                        MeanIndex = resultCount - 1
                    };
                    job.Schedule(dataSize, 256).Complete();
                }
            }

            taken.Dispose();
            partialSums.Dispose();
        }

        // For each data point, find cluster index that is closest to it
        [BurstCompile]
        private struct AssignClustersJob : IJobParallelFor
        {
            public int IndexOffset;
            public int Dim;
            [ReadOnly] public NativeArray<float> Data;
            [ReadOnly] public NativeArray<float> Means;
            [NativeDisableParallelForRestriction] public NativeArray<int> Clusters;
            [NativeDisableContainerSafetyRestriction] [NativeDisableParallelForRestriction] public NativeArray<float> Distances;

            public void Execute(int index)
            {
                index += IndexOffset;
                int meansCount = Means.Length / Dim;
                float minDist = float.MaxValue;
                int minIndex = 0;
                for (int i = 0; i < meansCount; ++i)
                {
                    float dist = DistanceSquared(Dim, Data, index, Means, i);
                    if (dist < minDist)
                    {
                        minIndex = i;
                        minDist = dist;
                    }
                }
                Clusters[index] = minIndex;
                if (Distances.IsCreated)
                    Distances[index] = minDist;
            }
        }

        private static void MakeRandomBatch(int dim, NativeArray<float> inputData, ref uint rngState, NativeArray<float> outBatch)
        {
            var job = new MakeBatchJob
            {
                Dim = dim,
                InputData = inputData,
                Seed = pcg_random(ref rngState),
                OutBatch = outBatch
            };
            job.Schedule().Complete();
        }

        [BurstCompile]
        private struct MakeBatchJob : IJob
        {
            public int Dim;
            public NativeArray<float> InputData;
            public NativeArray<float> OutBatch;
            public uint Seed;
            public void Execute()
            {
                uint dataSize = (uint)(InputData.Length / Dim);
                int batchSize = OutBatch.Length / Dim;
                NativeHashSet<int> picked = new(batchSize, Allocator.Temp);
                while (picked.Count < batchSize)
                {
                    int index = (int)(pcg_hash(Seed++) % dataSize);
                    if (!picked.Contains(index))
                    {
                        CopyElem(Dim, InputData, index, OutBatch, picked.Count);
                        picked.Add(index);
                    }
                }
                picked.Dispose();
            }
        }

        [BurstCompile]
        private struct UpdateCentroidsJob : IJob
        {
            public int Dim;
            public int BatchSize;
            [ReadOnly] public NativeArray<int> BatchClusters;
            public NativeArray<float> Counts;
            [ReadOnly] public NativeArray<float> BatchPoints;
            public NativeArray<float> Clusters;

            public void Execute()
            {
                for (int i = 0; i < BatchSize; ++i)
                {
                    int clusterIndex = BatchClusters[i];
                    Counts[clusterIndex]++;
                    float alpha = 1.0f / Counts[clusterIndex];

                    for (int j = 0; j < Dim; ++j)
                    {
                        Clusters[clusterIndex * Dim + j] = math.lerp(Clusters[clusterIndex * Dim + j],
                            BatchPoints[i * Dim + j], alpha);
                    }
                }
            }
        }

        private static bool InitializeCentroids(int dim, NativeArray<float> inputData, int initBatchSize, ref uint rngState, int initAttempts, NativeArray<float> outClusters, Func<float,bool> progress)
        {
            using var prof = s_profPlusPlus.Auto();

            int k = outClusters.Length / dim;
            int dataSize = inputData.Length / dim;
            initBatchSize = math.min(initBatchSize, dataSize);

            NativeArray<float> centroidBatch = new(initBatchSize * dim, Allocator.TempJob);
            NativeArray<float> validationBatch = new(initBatchSize * dim, Allocator.TempJob);
            MakeRandomBatch(dim, inputData, ref rngState, centroidBatch);
            MakeRandomBatch(dim, inputData, ref rngState, validationBatch);

            NativeArray<int> tmpIndices = new(initBatchSize, Allocator.TempJob);
            NativeArray<float> tmpDistances = new(initBatchSize, Allocator.TempJob);
            NativeArray<float> curCentroids = new(k * dim, Allocator.TempJob);

            float minDistSum = float.MaxValue;

            bool cancelled = false;
            for (int ia = 0; ia < initAttempts; ++ia)
            {
                if (progress != null && !progress((float) ia / initAttempts * 0.3f))
                {
                    cancelled = true;
                    break;
                }

                KMeansPlusPlus(dim, k, centroidBatch, curCentroids, tmpDistances, ref rngState);

                {
                    using var profPart = s_profAssignClusters.Auto();
                    AssignClustersJob job = new AssignClustersJob
                    {
                        Dim = dim,
                        Data = validationBatch,
                        Means = curCentroids,
                        IndexOffset = 0,
                        Clusters = tmpIndices,
                        Distances = tmpDistances
                    };
                    job.Schedule(initBatchSize, 1).Complete();
                }

                float distSum = 0;
                foreach (var d in tmpDistances)
                    distSum += d;

                // is this centroid better?
                if (distSum < minDistSum)
                {
                    minDistSum = distSum;
                    outClusters.CopyFrom(curCentroids);
                }
            }

            centroidBatch.Dispose();
            validationBatch.Dispose();
            tmpDistances.Dispose();
            tmpIndices.Dispose();
            curCentroids.Dispose();
            return !cancelled;
        }

        // https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
        private static uint pcg_hash(uint input)
        {
            uint state = input * 747796405u + 2891336453u;
            uint word = ((state >> (int)((state >> 28) + 4u)) ^ state) * 277803737u;
            return (word >> 22) ^ word;
        }

        private static float pcg_hash_float(uint input, float upTo)
        {
            uint val = pcg_hash(input);
            float f = math.asfloat(0x3f800000 | (val >> 9)) - 1.0f;
            return f * upTo;
        }

        private static uint pcg_random(ref uint rngState)
        {
            uint state = rngState;
            rngState = rngState * 747796405u + 2891336453u;
            uint word = ((state >> (int)((state >> 28) + 4u)) ^ state) * 277803737u;
            return (word >> 22) ^ word;
        }
    }
}
