using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace RotateCubes.BRGCube
{
    public static class BRGCubeUtility
    {
        public static readonly int kSizeOfMatrix = sizeof(float) * 4 * 4;
        public static readonly int kSizeOfPackedMatrix = sizeof(float) * 4 * 3;
        public static readonly int kSizeOfFloat4 = sizeof(float) * 4;
        public static readonly int kBytesPerInstance = kSizeOfPackedMatrix * 2 + kSizeOfFloat4;
        public static readonly int kExtraBytes = kSizeOfMatrix * 2;

        public static int ObjectToWorldNameId = Shader.PropertyToID("unity_ObjectToWorld");
        public static int WorldToObjectNameId = Shader.PropertyToID("unity_WorldToObject");
        public static int BaseColorNameId = Shader.PropertyToID("_BaseColor");
        
        public static int BufferCountForInstances(int bytesPerInstance, uint numInstances, int extraBytes = 0)
        {
            bytesPerInstance = (bytesPerInstance + sizeof(int) - 1) / sizeof(int) * sizeof(int);
            extraBytes = (extraBytes + sizeof(int) - 1) / sizeof(int) * sizeof(int);
            int totalBytes = (int) (bytesPerInstance * numInstances + extraBytes);
            return totalBytes / sizeof(int);
        }

        public static int BufferSizeForInstances(int bytesPerInstance, int numInstances, int alignment, int extraBytes = 0)
        {
            bytesPerInstance = (bytesPerInstance + alignment - 1) / alignment * alignment;
            extraBytes = (extraBytes + alignment - 1) / alignment * alignment;
            return bytesPerInstance * numInstances + extraBytes;
        }

        public static int BufferCount(int bufferSize, int stride)
        {
            return bufferSize / stride;
        }
        
        public static MetadataValue CreateMetadataValue(int nameID, int gpuAddress, bool isOverridden)
        {
            const uint kIsOverriddenBit = 0x80000000;
            return new MetadataValue
            {
                NameID = nameID,
                Value = (uint)gpuAddress | (isOverridden ? kIsOverriddenBit : 0),
            };
        }
        
        public static void PackedMatrices(this ref float4x4 trs, ref float4 packed1, ref float4 packed2, ref float4 packed3)
        {
            packed1[0] = trs[0][0];
            packed1[1] = trs[0][1];
            packed1[2] = trs[0][2];
            
            packed1[3] = trs[1][0];
            packed2[0] = trs[1][1];
            packed2[1] = trs[1][2];
            
            packed2[2] = trs[2][0];
            packed2[3] = trs[2][1];
            packed3[0] = trs[2][2];
            
            packed3[1] = trs[3][0];
            packed3[2] = trs[3][1];
            packed3[3] = trs[3][2];
        }
        
        public static unsafe T* Malloc<T>(int count) where T : unmanaged
        {
            return (T*)UnsafeUtility.Malloc(
                UnsafeUtility.SizeOf<T>() * count,
                UnsafeUtility.AlignOf<T>(),
                Allocator.TempJob);
        }
    }
}