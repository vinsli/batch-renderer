using Unity.Burst;
using Unity.Mathematics;
using UnityEngine;

namespace RotateCubes.BRGCube.MovingCubesMultiBatches
{
    [BurstCompile]
    public struct SpawnUtilities
    {
        public static float4 ComputeColor(int index, int maxCount, float saturation = 1)
        {
            float t = (float) index / math.max(1, maxCount - 1);
            var color = Color.HSVToRGB(t, saturation, 1);
            return new float4(color.r, color.g, color.b, 1);
        }

        public static float3 ComputePosition(int index, int2 dim, float3 origin, float3 scale)
        {
            int x = index % dim.y;
            int y = index / dim.y;

            float2 uv = new float2(
                (float) x / (dim.x - 1),
                (float) y / (dim.y - 1));

            float3 extent = new float3(scale.x, 0, scale.z) / 2.0f;
            float3 min = origin - extent;
            float3 max = origin + extent;
            float3 pos = new float3(
                math.lerp(min.x, max.x, uv.x),
                origin.y,
                math.lerp(min.z, max.z, uv.y));
            return pos;
        }

        public static float4x4 ComputeTransform(float3 pos, float scale)
        {
            float4x4 transform = float4x4.TRS(pos, quaternion.identity, scale);
            return transform;
        }
    }
}