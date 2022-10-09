using System;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngine.Rendering;
using static RotateCubes.BRGCube.BRGCubeUtility;
using FrustumPlanes = Unity.Rendering.FrustumPlanes.FrustumPlanes;
using Random = Unity.Mathematics.Random;

namespace RotateCubes.BRGCube.MovingCubesMultiBatches
{
    public class MovingCubesMultiBatches : MonoBehaviour
    {
        private struct DrawKey : IEquatable<DrawKey>, IComparable<DrawKey>
        {
            public BatchMeshID MeshID;
            public uint SubmeshIndex;
            public BatchMaterialID MaterialID;

            public override int GetHashCode()
            {
                return HashCode.Combine(MeshID, SubmeshIndex, MaterialID);
            }

            public int CompareTo(DrawKey other)
            {
                int cmpMaterial = MaterialID.CompareTo(other.MaterialID);
                int cmpMesh = MeshID.CompareTo(other.MeshID);
                int cmpSubmesh = SubmeshIndex.CompareTo(other.SubmeshIndex);

                if (cmpMaterial != 0)
                    return cmpMaterial;
                if (cmpMesh != 0)
                    return cmpMesh;

                return cmpSubmesh;
            }

            public bool Equals(DrawKey other) => CompareTo(other) == 0;
        }
        
        private struct SrpBatch
        {
            public BatchID BatchID;
            public DrawKey DrawKey;
            public int GraphicsBufferOffsetInFloat4;
            public int InstanceOffset;
            public int InstanceCount;
        }
        
        private struct SrpBatchDrawCommand
        {
            public int VisibleOffset;
            public int VisibleCount;
        }

        public int instanceCount;
        public float rotateSpeed;
        public float moveSpeed;

        public List<Material> materials;
        public List<Mesh> meshes;

        private BatchRendererGroup _brg;
        private List<BatchMeshID> _meshIds;
        private List<BatchMaterialID> _materialIds;
        private List<BatchID> _batchIds;

        private Dictionary<DrawKey, NativeList<int>> _batchesPerDrawKey = new();

        private Dictionary<DrawKey, NativeList<float3>> _positionsPerDrawKey = new ();
        private Dictionary<DrawKey, NativeList<quaternion>> _rotationsPerDrawKey = new ();
        private Dictionary<DrawKey, NativeList<AABB>> _aabbsPerDrawKey = new();

        private Dictionary<DrawKey, NativeArray<float4>> _objectToWorldPerDrawKey = new ();
        private Dictionary<DrawKey, NativeArray<float4>> _worldToObjectPerDrawKey = new ();
        private Dictionary<DrawKey, NativeList<float4>> _colorsPerDrawKey = new ();
        private Dictionary<DrawKey, GraphicsBuffer> _instanceDataPerDrawKey = new();
        private Dictionary<DrawKey, NativeQueue<BatchDrawCommand>> _batchDrawCommandsPerDrawKey = new();

        private GraphicsBuffer _globals;

        private int _maxItemPerBatch;
        private NativeList<SrpBatch> _drawBatches;
        private NativeArray<float4> _brgHeader;
        
        private Random _random = new Random(83729);

        private float3 _rootPos;
        private float _startTime;

        private JobHandle _uploadInstanceDataHandle;
        
        private void Start()
        {
            _drawBatches = new NativeList<SrpBatch>(128, Allocator.Persistent);
            _brgHeader = new NativeArray<float4>(4, Allocator.Persistent);
            for (int i = 0; i < 4; i++)
            {
                _brgHeader[i] = float4.zero;
            }
            
            _rootPos = transform.position;
            _brg = new BatchRendererGroup(OnPerformCulling, IntPtr.Zero);
            _meshIds = new List<BatchMeshID>(meshes.Count);
            _materialIds = new List<BatchMaterialID>(materials.Count);

            for (int i = 0; i < meshes.Count; i++)
            {
                _meshIds.Add(_brg.RegisterMesh(meshes[i]));
            }

            for (int i = 0; i < materials.Count; i++)
            {
                _materialIds.Add(_brg.RegisterMaterial(materials[i]));
            }

            //global defaults
            _globals = new GraphicsBuffer(GraphicsBuffer.Target.Constant, 1, UnsafeUtility.SizeOf<BatchRendererGroupGlobals>());
            _globals.SetData(new[] {BatchRendererGroupGlobals.Default});

            GenerateRandomInstanceData();
            // GenerateInstanceData();
            GenerateBatches();
            InitializeBatchHeader();
            
            var updateInstanceDataJobHandle = UpdateInstanceData();
            UploadInstanceData(updateInstanceDataJobHandle);
            
            Bounds bounds = new Bounds(new Vector3(0, 0, 0), new Vector3(1048576.0f, 1048576.0f, 1048576.0f));
            _brg.SetGlobalBounds(bounds);
            _startTime = Time.realtimeSinceStartup;
        }
        
        private void OnDestroy()
        {
            foreach (var batchKey in _colorsPerDrawKey.Keys)
            {
                _colorsPerDrawKey[batchKey].Dispose();
                _positionsPerDrawKey[batchKey].Dispose();
                _rotationsPerDrawKey[batchKey].Dispose();
                _aabbsPerDrawKey[batchKey].Dispose();
                _worldToObjectPerDrawKey[batchKey].Dispose();
                _objectToWorldPerDrawKey[batchKey].Dispose();
                _instanceDataPerDrawKey[batchKey].Dispose();
                _batchesPerDrawKey[batchKey].Dispose();
                _batchDrawCommandsPerDrawKey[batchKey].Dispose();
            }

            _drawBatches.Dispose();
            _brgHeader.Dispose();
            _brg.Dispose();
            _globals.Dispose();
        }
        
        private void GenerateBatches()
        {
            int kBRGBufferMaxWindowSize = 16 * 1024 * 1024;
            const int kItemSize = (2 * 3 + 1);  //  size in "float4" ( 2 * 4x3 matrices plus 1 color per item )
            _maxItemPerBatch = ((kBRGBufferMaxWindowSize / kSizeOfFloat4) - 4) / kItemSize;  // -4 "float4" for 64 first 0 bytes ( BRG contrainst )
            // if (_maxItemPerBatch > instanceCount)
            //     _maxItemPerBatch = instanceCount;
            
            foreach (var drawKey in _colorsPerDrawKey.Keys)
            {
                if (!_batchesPerDrawKey.ContainsKey(drawKey))
                {
                    _batchesPerDrawKey.Add(drawKey, new NativeList<int>(128, Allocator.Persistent));
                }
                
                var instanceCountPerDrawKey = _colorsPerDrawKey[drawKey].Length;
                _worldToObjectPerDrawKey.Add(drawKey, new NativeArray<float4>(instanceCountPerDrawKey * 3, Allocator.Persistent));
                _objectToWorldPerDrawKey.Add(drawKey, new NativeArray<float4>(instanceCountPerDrawKey * 3, Allocator.Persistent));

                var maxItemPerDrawKeyBatch = _maxItemPerBatch > instanceCountPerDrawKey ? instanceCountPerDrawKey : _maxItemPerBatch;
                //gather batch count per drawkey
                int batchAlignedSizeInFloat4 = BufferSizeForInstances(kBytesPerInstance, maxItemPerDrawKeyBatch, kSizeOfFloat4, 4 * kSizeOfFloat4) / kSizeOfFloat4;
                var batchCountPerDrawKey = (instanceCountPerDrawKey + maxItemPerDrawKeyBatch - 1) / maxItemPerDrawKeyBatch;
                
                //create instance data buffer
                var instanceDataCountInFloat4 =  batchCountPerDrawKey * batchAlignedSizeInFloat4;
                var target = GraphicsBuffer.Target.Raw;
                var instanceData = new GraphicsBuffer(target, GraphicsBuffer.UsageFlags.LockBufferForWrite, instanceDataCountInFloat4, kSizeOfFloat4);
                _instanceDataPerDrawKey.Add(drawKey, instanceData);
                
                //generate srp batches
                int left = instanceCountPerDrawKey;
                for (int i = 0; i < batchCountPerDrawKey; i++)
                {
                    int instanceOffset = i * maxItemPerDrawKeyBatch;
                    int gpuOffsetInFloat4 = i * batchAlignedSizeInFloat4;
                    
                    var batchInstanceCount = left > maxItemPerDrawKeyBatch ? maxItemPerDrawKeyBatch : left;
                    var drawBatch = new SrpBatch
                    {
                        DrawKey = drawKey,
                        GraphicsBufferOffsetInFloat4 = gpuOffsetInFloat4,
                        InstanceOffset = instanceOffset,
                        InstanceCount = batchInstanceCount
                    };
                    
                    _batchesPerDrawKey[drawKey].Add(_drawBatches.Length);
                    _drawBatches.Add(drawBatch);
                    left -= batchInstanceCount;
                }
            }
            
            int objectToWorldID = Shader.PropertyToID("unity_ObjectToWorld");
            int worldToObjectID = Shader.PropertyToID("unity_WorldToObject");
            int colorID = Shader.PropertyToID("_BaseColor");
            var batchMetadata = new NativeArray<MetadataValue>(3, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            for (int i = 0; i < _drawBatches.Length; i++)
            {
                var drawBatch = _drawBatches[i];
                var instanceData = _instanceDataPerDrawKey[drawBatch.DrawKey];
                
                var baseOffset = drawBatch.GraphicsBufferOffsetInFloat4 * kSizeOfFloat4;
                batchMetadata[0] = CreateMetadataValue(objectToWorldID, baseOffset + 64, true);       // matrices
                batchMetadata[1] = CreateMetadataValue(worldToObjectID, baseOffset + 64 + kSizeOfPackedMatrix * drawBatch.InstanceCount, true); // inverse matrices
                batchMetadata[2] = CreateMetadataValue(colorID, baseOffset + 64 + kSizeOfPackedMatrix * drawBatch.InstanceCount * 2, true); // colors
                
                drawBatch.BatchID = _brg.AddBatch(batchMetadata, instanceData.bufferHandle, 0, 0);
                _drawBatches[i] = drawBatch;
            }
        }

        private void GenerateRandomInstanceData()
        {
            var dim = new int2(math.ceil(math.sqrt(instanceCount)));
            
            for (int i = 0; i < instanceCount; i++)
            {
                var meshIdx = _random.NextInt(_meshIds.Count);
                var drawKey = new DrawKey
                {
                    MaterialID = _materialIds[_random.NextInt(_materialIds.Count)],
                    MeshID = _meshIds[meshIdx],
                    SubmeshIndex = 0
                };

                NativeList<float4> colors;
                NativeList<float3> positions;
                NativeList<quaternion> rotations;
                NativeList<AABB> aabbs;
                if (!_colorsPerDrawKey.ContainsKey(drawKey))
                {
                    colors = new NativeList<float4>(2048, Allocator.Persistent);
                    positions = new NativeList<float3>(2048, Allocator.Persistent);
                    rotations = new NativeList<quaternion>(2048, Allocator.Persistent);
                    aabbs = new NativeList<AABB>(2048, Allocator.Persistent);
                    _colorsPerDrawKey.Add(drawKey, colors);
                    _positionsPerDrawKey.Add(drawKey, positions);
                    _rotationsPerDrawKey.Add(drawKey, rotations);
                    _aabbsPerDrawKey.Add(drawKey, aabbs);
                    
                    NativeQueue<BatchDrawCommand> batchDrawCommands = new NativeQueue<BatchDrawCommand>(Allocator.Persistent);
                    _batchDrawCommandsPerDrawKey.Add(drawKey, batchDrawCommands);
                }
                else
                {
                    colors = _colorsPerDrawKey[drawKey];
                    positions = _positionsPerDrawKey[drawKey];
                    rotations = _rotationsPerDrawKey[drawKey];
                    aabbs = _aabbsPerDrawKey[drawKey];
                }

                colors.Add(SpawnUtilities.ComputeColor(i, instanceCount));
                
                positions.Add(SpawnUtilities.ComputePosition(i, dim, _rootPos, new float3(100, 1, 100)));
                rotations.Add(_random.NextQuaternionRotation());
                aabbs.Add(meshes[meshIdx].bounds.ToAABB());
            }
        }

        private void GenerateInstanceData()
        {
            const int instanceCountPerDrawKey = 100;
            instanceCount = 0;
            
            var drawKey0 = new DrawKey
            {
                MaterialID = _materialIds[0],
                MeshID = _meshIds[0],
                SubmeshIndex = 0
            };

            var colors = new NativeList<float4>(instanceCountPerDrawKey, Allocator.Persistent);
            var positions = new NativeList<float3>(instanceCountPerDrawKey, Allocator.Persistent);
            var rotations = new NativeList<quaternion>(instanceCountPerDrawKey, Allocator.Persistent);
            var aabbs = new NativeList<AABB>(instanceCountPerDrawKey, Allocator.Persistent);
            var batchDrawCommands = new NativeQueue<BatchDrawCommand>(Allocator.Persistent);
            _colorsPerDrawKey.Add(drawKey0, colors);
            _positionsPerDrawKey.Add(drawKey0, positions);
            _rotationsPerDrawKey.Add(drawKey0, rotations);
            _aabbsPerDrawKey.Add(drawKey0, aabbs);
            _batchDrawCommandsPerDrawKey.Add(drawKey0, batchDrawCommands);

            for (int i = 0; i < instanceCountPerDrawKey; i++)
            {
                colors.Add(new float4(_random.NextFloat3(), 1));

                var randPos = _random.NextFloat3Direction() * math.pow(_random.NextFloat(0.1f, 1), 1f / 3f);
                positions.Add(randPos * 10 + _rootPos);
                rotations.Add(_random.NextQuaternionRotation());
                aabbs.Add(meshes[0].bounds.ToAABB());
            }
            
            instanceCount += instanceCountPerDrawKey;
            
            //second drawkey
            var drawKey1 = new DrawKey
            {
                MaterialID = _materialIds[1],
                MeshID = _meshIds[1],
                SubmeshIndex = 0
            };
            
            colors = new NativeList<float4>(instanceCountPerDrawKey, Allocator.Persistent);
            positions = new NativeList<float3>(instanceCountPerDrawKey, Allocator.Persistent);
            rotations = new NativeList<quaternion>(instanceCountPerDrawKey, Allocator.Persistent);
            aabbs = new NativeList<AABB>(instanceCountPerDrawKey, Allocator.Persistent);
            batchDrawCommands = new NativeQueue<BatchDrawCommand>(Allocator.Persistent);
            _colorsPerDrawKey.Add(drawKey1, colors);
            _positionsPerDrawKey.Add(drawKey1, positions);
            _rotationsPerDrawKey.Add(drawKey1, rotations);
            _aabbsPerDrawKey.Add(drawKey1, aabbs);
            _batchDrawCommandsPerDrawKey.Add(drawKey1, batchDrawCommands);
            
            for (int i = 0; i < instanceCountPerDrawKey; i++)
            {
                colors.Add(new float4(_random.NextFloat3(), 1));
            
                var randPos = _random.NextFloat3Direction() * math.pow(_random.NextFloat(0, 1), 1f / 3f);
                positions.Add(randPos * 10 + _rootPos);
                rotations.Add(_random.NextQuaternionRotation());
                aabbs.Add(meshes[1].bounds.ToAABB());
            }
            
            instanceCount += instanceCountPerDrawKey;
        }

        private void Update()
        {
            Shader.SetGlobalConstantBuffer(BatchRendererGroupGlobals.kGlobalsPropertyId, _globals, 0, _globals.stride);

            JobHandle updateInstanceDataJobHandle = default;
            Profiler.BeginSample("UpdateInstanceData");
            updateInstanceDataJobHandle = UpdateInstanceData();
            Profiler.EndSample();
            
            Profiler.BeginSample("UploadInstanceData");
            UploadInstanceData(updateInstanceDataJobHandle);
            Profiler.EndSample();
        }
        
        private void InitializeBatchHeader()
        {
            foreach (var srpBatch in _drawBatches)
            {
                var instanceData = _instanceDataPerDrawKey[srpBatch.DrawKey];
                instanceData.SetData(_brgHeader, 0, srpBatch.GraphicsBufferOffsetInFloat4, _brgHeader.Length);
            }
        }

        private void UploadInstanceData(JobHandle updateInstanceDataJobHandle)
        {
            // UploadInstanceDataMainThread(updateInstanceDataJobHandle);
            UploadInstanceDataJobs(updateInstanceDataJobHandle);
        }

        private void UploadInstanceDataMainThread(JobHandle updateInstanceDataJobHandle)
        {
            updateInstanceDataJobHandle.Complete();
            // var colors = _colorsPerDrawKey[drawKey];
            foreach (var pair in _batchesPerDrawKey)
            {
                var drawKey = pair.Key;
                var instanceData = _instanceDataPerDrawKey[drawKey];
                var batches = pair.Value;
                var objectToWorldMatrices = _objectToWorldPerDrawKey[drawKey];
                var worldToObjectMatrices = _worldToObjectPerDrawKey[drawKey];
                var colors = _colorsPerDrawKey[drawKey];

                foreach (var srpBatchIdx in batches)
                {
                    var srpBatch = _drawBatches[srpBatchIdx];
                    var batchInstanceCount = srpBatch.InstanceCount;
                    var cpuBufferOffset = srpBatch.InstanceOffset;
                
                    var objectToWorldGpuBufferOffset = srpBatch.GraphicsBufferOffsetInFloat4 + 4;
                    var worldToObjectGpuBufferOffset = objectToWorldGpuBufferOffset + batchInstanceCount * 3;
                    var colorGpuBufferOffset = worldToObjectGpuBufferOffset + batchInstanceCount * 3;
                
                    instanceData.SetData(objectToWorldMatrices, cpuBufferOffset * 3, objectToWorldGpuBufferOffset, batchInstanceCount * 3);
                    instanceData.SetData(worldToObjectMatrices, cpuBufferOffset * 3, worldToObjectGpuBufferOffset, batchInstanceCount * 3);
                    instanceData.SetData((NativeArray<float4>)colors, cpuBufferOffset, colorGpuBufferOffset, batchInstanceCount);
                }
            }
        }

        private void UploadInstanceDataJobs(JobHandle updateInstanceDataJobHandle)
        {
            NativeList<JobHandle> handles = new NativeList<JobHandle>(_batchesPerDrawKey.Count * 2, Allocator.Temp);
            
            foreach (var pair in _batchesPerDrawKey)
            {
                var drawKey = pair.Key;
                var instanceData = _instanceDataPerDrawKey[drawKey];
                var batchIds = pair.Value;
                var objectToWorldMatrices = _objectToWorldPerDrawKey[drawKey];
                var worldToObjectMatrices = _worldToObjectPerDrawKey[drawKey];
                var colors = _colorsPerDrawKey[drawKey];

                handles.Add(new UploadInstanceDataJob
                {
                    ObjectToWorldMatrices = objectToWorldMatrices,
                    WorldToObjectMatrices = worldToObjectMatrices,
                    Colors = colors,
                    BatchIds = batchIds,
                    Batches = _drawBatches,
                    Output = instanceData.LockBufferForWrite<float4>(0, instanceData.count)
                }.Schedule(updateInstanceDataJobHandle));
            }
            
            _uploadInstanceDataHandle = JobHandle.CombineDependencies(handles);
            _uploadInstanceDataHandle.Complete();
            
            foreach (var pair in _instanceDataPerDrawKey)
            {
                var instanceData = pair.Value;
                instanceData.UnlockBufferAfterWrite<float4>(instanceData.count);
            }
        }
        
        [BurstCompile]
        private struct UploadInstanceDataJob : IJob
        {
            [ReadOnly]
            public NativeArray<float4> ObjectToWorldMatrices;
            [ReadOnly]
            public NativeArray<float4> WorldToObjectMatrices;
            [ReadOnly]
            public NativeArray<float4> Colors;
            [ReadOnly]
            public NativeList<int> BatchIds;
            [ReadOnly]
            public NativeList<SrpBatch> Batches;
            
            [WriteOnly]
            public NativeArray<float4> Output;
            public void Execute()
            {
                int batchCount = BatchIds.Length;
                for (int i = 0; i < batchCount; i++)
                {
                    var srpBatch = Batches[BatchIds[i]];
                    
                    var batchInstanceCount = srpBatch.InstanceCount;
                    var cpuBufferOffset = srpBatch.InstanceOffset;
                    
                    var objectToWorldGpuBufferOffset = srpBatch.GraphicsBufferOffsetInFloat4 + 4;
                    var worldToObjectGpuBufferOffset = objectToWorldGpuBufferOffset + batchInstanceCount * 3;
                    var colorGpuBufferOffset = worldToObjectGpuBufferOffset + batchInstanceCount * 3;
                    
                    for (int j = 0; j < batchInstanceCount; j++)
                    {
                        var matrixOffset = j * 3;
                        var matrixCpuBufferOffset = cpuBufferOffset * 3 + matrixOffset;
                        
                        Output[objectToWorldGpuBufferOffset + matrixOffset] = ObjectToWorldMatrices[matrixCpuBufferOffset];
                        Output[objectToWorldGpuBufferOffset + matrixOffset + 1] = ObjectToWorldMatrices[matrixCpuBufferOffset + 1];
                        Output[objectToWorldGpuBufferOffset + matrixOffset + 2] = ObjectToWorldMatrices[matrixCpuBufferOffset + 2];
                        
                        Output[worldToObjectGpuBufferOffset + matrixOffset] = WorldToObjectMatrices[matrixCpuBufferOffset];
                        Output[worldToObjectGpuBufferOffset + matrixOffset + 1] = WorldToObjectMatrices[matrixCpuBufferOffset + 1];
                        Output[worldToObjectGpuBufferOffset + matrixOffset + 2] = WorldToObjectMatrices[matrixCpuBufferOffset + 2];
                        
                        Output[colorGpuBufferOffset + j] = Colors[cpuBufferOffset + j];
                    }
                }
            }
        }
        
        [BurstCompile]
        private unsafe struct UpdateInstanceDataJob : IJobFor
        {
            public NativeArray<float3> Positions;
            public NativeArray<quaternion> Rotations;

            [NativeDisableParallelForRestriction]
            public NativeArray<float4> ObjectToWorldMatrices;
            [NativeDisableParallelForRestriction]
            public NativeArray<float4> WorldToObjectMatrices;

            public float3 Origin;
            
            public float RotateSpeed;
            public float MoveSpeed;
            public float DeltaTime;
            public float ElapsedTime;
            
            public void Execute(int i)
            {
                var positionsPtr = (float3*) Positions.GetUnsafeReadOnlyPtr();
                var objectToWorldPtr = (float4*) ObjectToWorldMatrices.GetUnsafePtr();
                var worldToObjectPtr = (float4*) WorldToObjectMatrices.GetUnsafePtr();

                UpdatePosition(ref positionsPtr[i], ref Origin, MoveSpeed, ElapsedTime, out var newPosition);
                // var newPosition = positionsPtr[i];
                var rot = math.mul(Rotations[i], quaternion.RotateY(math.radians(DeltaTime * RotateSpeed)));
                Rotations[i] = rot;

                var matrixIdx = i * 3;
                var objectToWorldMatrix = float4x4.TRS(newPosition, rot, new float3(1, 1, 1));
                objectToWorldMatrix.PackedMatrices(ref objectToWorldPtr[matrixIdx], ref objectToWorldPtr[matrixIdx + 1], ref objectToWorldPtr[matrixIdx + 2]);
                    
                var worldToObjectMatrix = math.inverse(objectToWorldMatrix);
                worldToObjectMatrix.PackedMatrices(ref worldToObjectPtr[matrixIdx], ref worldToObjectPtr[matrixIdx + 1], ref worldToObjectPtr[matrixIdx + 2]);
            }

            private void UpdatePosition(ref float3 startPosition, ref float3 origin, float speed, float elapsedTime, out float3 newPosition)
            {
                var relativeStart = startPosition - origin;

                var p0 = relativeStart.xz;

                var r = math.length(p0);

                var angle0 = math.atan2(p0.y, p0.x);
                var angularSpeed = speed / r;
                var angle = (float) math.fmod(angle0 + angularSpeed * elapsedTime, math.PI_DBL * 2);
                var p = new float2(math.cos(angle), math.sin(angle)) * r;
                newPosition =  new float3(p.x, origin.y, p.y);
            }
        }

        private JobHandle UpdateInstanceData()
        {
            NativeList<JobHandle> handles = new NativeList<JobHandle>(_positionsPerDrawKey.Count * 2, Allocator.Temp);
            
            foreach (var pair in _positionsPerDrawKey)
            {
                var drawKey = pair.Key;
                var positions = pair.Value;
                var rotations = _rotationsPerDrawKey[drawKey];
                var objectToWorldMatrices = _objectToWorldPerDrawKey[drawKey];
                var worldToObjectMatrices = _worldToObjectPerDrawKey[drawKey];
                var colors = _colorsPerDrawKey[drawKey];
                
                handles.Add(new UpdateInstanceDataJob
                {
                    Positions = positions,
                    Rotations = rotations,
                    ObjectToWorldMatrices = objectToWorldMatrices,
                    WorldToObjectMatrices = worldToObjectMatrices,
                    
                    Origin = _rootPos,
                    
                    RotateSpeed = rotateSpeed,
                    MoveSpeed = moveSpeed,
                    DeltaTime = Time.deltaTime,
                    ElapsedTime = Time.realtimeSinceStartup - _startTime,
                }.ScheduleParallel(positions.Length, 64, new JobHandle()));
            }
            
            JobHandle.ScheduleBatchedJobs();
            return JobHandle.CombineDependencies(handles);
        }

        public unsafe JobHandle OnPerformCulling(
            BatchRendererGroup rendererGroup,
            BatchCullingContext cullingContext,
            BatchCullingOutput cullingOutput,
            IntPtr userContext)
        {
            BatchCullingOutputDrawCommands drawCommands = new BatchCullingOutputDrawCommands();
            
            // CullingMainThread(ref drawCommands, cullingContext.cullingPlanes);
            
            CullingThreaded(ref drawCommands, cullingContext.cullingPlanes);
            
            cullingOutput.drawCommands[0] = drawCommands;
            return new JobHandle();
        }
        
        [BurstCompile]
        private unsafe struct CullingJob : IJobFor
        {
            [ReadOnly]
            public NativeArray<Plane> CullingPlanes;
            [DeallocateOnJobCompletion]
            [ReadOnly]
            public NativeArray<SrpBatch> Batches;
            [ReadOnly]
            public NativeArray<AABB> LocalAABB;
            [ReadOnly]
            [NativeDisableContainerSafetyRestriction]
            public NativeArray<float4> ObjectToWorldMatrices;
            [ReadOnly]
            public int DrawKeyOffset;
            
            [WriteOnly]
            [NativeDisableUnsafePtrRestriction]
            public int* VisibleInstances;
            [WriteOnly]
            public NativeQueue<BatchDrawCommand>.ParallelWriter DrawCommands;

            public void Execute(int index)
            {
                var batchesPtr = (SrpBatch *)Batches.GetUnsafeReadOnlyPtr();
                var localAABBPtr = (AABB*) LocalAABB.GetUnsafeReadOnlyPtr();
                var objectToWorldMatricesPtr = (float4 *)ObjectToWorldMatrices.GetUnsafeReadOnlyPtr();
                ref var srpBatch = ref batchesPtr[index];
                int visibleCount = 0;
                int batchOffset = DrawKeyOffset + srpBatch.InstanceOffset;
                for (int instanceIdx = 0; instanceIdx < srpBatch.InstanceCount; instanceIdx++)
                {
                    //Assume only have 1 culling split and have 6 culling planes
                    ref var localAABB = ref localAABBPtr[srpBatch.InstanceOffset + instanceIdx];
                    var matrixIdx = (srpBatch.InstanceOffset + instanceIdx) * 3;
                    var worldAABB = Transform(ref objectToWorldMatricesPtr[matrixIdx], ref objectToWorldMatricesPtr[matrixIdx + 1], ref objectToWorldMatricesPtr[matrixIdx + 2], ref localAABB);
                    if (Intersect(CullingPlanes, ref worldAABB) == FrustumPlanes.IntersectResult.Out)
                        continue;
                    
                    VisibleInstances[batchOffset + visibleCount] = instanceIdx;
                    visibleCount++;
                }

                if (visibleCount > 0)
                {
                    var drawKey = srpBatch.DrawKey;
                    DrawCommands.Enqueue(new BatchDrawCommand
                    {
                        visibleOffset = (uint) batchOffset,
                        visibleCount = (uint) visibleCount,
                        batchID = srpBatch.BatchID,
                        materialID = drawKey.MaterialID,
                        meshID = drawKey.MeshID,
                        submeshIndex = (ushort) drawKey.SubmeshIndex,
                        splitVisibilityMask = 0xff,
                        flags = BatchDrawCommandFlags.None,
                        sortingPosition = 0
                    });
                }
            }
        }

        private unsafe void CullingThreaded(ref BatchCullingOutputDrawCommands drawCommands, NativeArray<Plane> cullingPlanes)
        {
            drawCommands.visibleInstances = Malloc<int>(instanceCount);
            NativeArray<JobHandle> cullingJobs = new NativeArray<JobHandle>(_batchesPerDrawKey.Count, Allocator.Temp);

            int drawKeyIdx = 0;
            int totalInstanceCount = 0;
            foreach (var pair in _batchesPerDrawKey)
            {
                int instanceCountPerDrawKey = 0;
                foreach (var batchIdx in pair.Value)
                {
                    instanceCountPerDrawKey += _drawBatches[batchIdx].InstanceCount;
                }
                
                var drawKey = pair.Key;
                var batchIds = pair.Value;
                var batchDrawCommands = _batchDrawCommandsPerDrawKey[drawKey];

                var batchesPerDrawKey = new NativeArray<SrpBatch>(batchIds.Length, Allocator.TempJob);
                for (int i = 0; i < batchIds.Length; i++)
                {
                    batchesPerDrawKey[i] = _drawBatches[batchIds[i]];
                }
                
                cullingJobs[drawKeyIdx] = new CullingJob
                {
                    CullingPlanes = cullingPlanes,
                    Batches = batchesPerDrawKey,
                    LocalAABB = _aabbsPerDrawKey[drawKey],
                    DrawKeyOffset = totalInstanceCount,
                    ObjectToWorldMatrices = _objectToWorldPerDrawKey[drawKey],
                
                    VisibleInstances = drawCommands.visibleInstances,
                    DrawCommands = batchDrawCommands.AsParallelWriter(),
                }.ScheduleParallel(batchIds.Length, 32, _uploadInstanceDataHandle);
                
                // for (int i = 0; i < batchIds.Length; i++)
                // {
                //     new CullingJob
                //     {
                //         CullingPlanes = cullingPlanes,
                //         Batches = batchesPerDrawKey,
                //         LocalAABB = _aabbsPerDrawKey[drawKey],
                //         DrawKeyOffset = totalInstanceCount,
                //         ObjectToWorldMatrices = _objectToWorldPerDrawKey[drawKey],
                //
                //         VisibleInstances = drawCommands.visibleInstances,
                //         DrawCommands = batchDrawCommands.AsParallelWriter(),
                //     }.Execute(i);
                // }
                
                totalInstanceCount += instanceCountPerDrawKey;
                drawKeyIdx++;
            }
            
            JobHandle.CombineDependencies(cullingJobs).Complete();
            
            //count total draw commands
            var totalBatchDrawCommands = 0;
            foreach (var pair in _batchDrawCommandsPerDrawKey)
            {
                totalBatchDrawCommands += pair.Value.Count;
            }
            
            SetupDrawRanges(ref drawCommands, totalBatchDrawCommands);
            
            drawCommands.instanceSortingPositions = null;
            drawCommands.instanceSortingPositionFloatCount = 0;
            
            drawCommands.drawCommandCount = totalBatchDrawCommands;
            drawCommands.drawCommands = Malloc<BatchDrawCommand>(drawCommands.drawCommandCount);

            int drawCommandIdx = 0;
            foreach (var drawCommandsQueue in _batchDrawCommandsPerDrawKey.Values)
            {
                while (drawCommandsQueue.TryDequeue(out var drawCommand))
                {
                    drawCommands.drawCommands[drawCommandIdx] = drawCommand;
                    drawCommandIdx++;
                }
            }
        }

        private unsafe void CullingMainThread(ref BatchCullingOutputDrawCommands drawCommands, NativeArray<Plane> cullingPlanes)
        {
            var batchCount = _drawBatches.Length;
            drawCommands.visibleInstances = Malloc<int>(instanceCount);
            NativeList<BatchDrawCommand> batchDrawCommands = new NativeList<BatchDrawCommand>(batchCount, Allocator.Temp);

            int visibleOffset = 0;
            for (int batchIdx = 0; batchIdx < batchCount; batchIdx++)
            {
                int visibleCount = 0;
                var srpBatch = _drawBatches[batchIdx];
                var drawKey = srpBatch.DrawKey;
                var aabbs = (AABB *)_aabbsPerDrawKey[drawKey].GetUnsafeReadOnlyPtr();
                var objectToWorldMatrices = (float4 *) _objectToWorldPerDrawKey[drawKey].GetUnsafeReadOnlyPtr();
                for (int instanceIdx = 0; instanceIdx < srpBatch.InstanceCount; instanceIdx++)
                {
                    //Assume only have 1 culling split and have 6 culling planes
                    ref var localAABB = ref aabbs[srpBatch.InstanceOffset + instanceIdx];
                    var matrixIdx = (srpBatch.InstanceOffset + instanceIdx) * 3;
                    var worldAABB = Transform(ref objectToWorldMatrices[matrixIdx], ref objectToWorldMatrices[matrixIdx + 1], ref objectToWorldMatrices[matrixIdx + 2], ref localAABB);
                    if (Intersect(cullingPlanes, ref worldAABB) == FrustumPlanes.IntersectResult.Out)
                        continue;
                    
                    drawCommands.visibleInstances[visibleOffset + visibleCount] = instanceIdx;
                    visibleCount++;
                }
                
                // Debug.Log($"[{Time.frameCount}]batch[{batchIdx}] visible instance count: [{visibleCount}]");

                if (visibleCount > 0)
                {
                    batchDrawCommands.Add(new BatchDrawCommand
                    {
                        visibleOffset = (uint) visibleOffset,
                        visibleCount = (uint) visibleCount,
                        batchID = srpBatch.BatchID,
                        materialID = drawKey.MaterialID,
                        meshID = drawKey.MeshID,
                        submeshIndex = (ushort) drawKey.SubmeshIndex,
                        splitVisibilityMask = 0xff,
                        flags = BatchDrawCommandFlags.None,
                        sortingPosition = 0
                    });
                }
                
                visibleOffset += visibleCount;
            }
            
            drawCommands.drawCommandCount = batchDrawCommands.Length;
            drawCommands.drawCommands = Malloc<BatchDrawCommand>(drawCommands.drawCommandCount);
            for (int i = 0; i < batchDrawCommands.Length; i++)
            {
                drawCommands.drawCommands[i] = batchDrawCommands[i];
            }
            
            SetupDrawRanges(ref drawCommands, batchDrawCommands.Length);
            
            drawCommands.instanceSortingPositions = null;
            drawCommands.instanceSortingPositionFloatCount = 0;
        }

        private unsafe void SetupDrawRanges(ref BatchCullingOutputDrawCommands drawCommands, int drawCommandsCount)
        {
            //Assume only have 1 draw range
            drawCommands.drawRangeCount = 1;
            drawCommands.drawRanges = Malloc<BatchDrawRange>(1);
            drawCommands.drawRanges[0] = new BatchDrawRange
            {
                drawCommandsBegin = 0,
                drawCommandsCount = (uint) drawCommandsCount,
                filterSettings = new BatchFilterSettings
                {
                    renderingLayerMask = 1,
                    layer = 0,
                    motionMode = MotionVectorGenerationMode.Camera,
                    shadowCastingMode = ShadowCastingMode.Off,
                    receiveShadows = false,
                    staticShadowCaster = false,
                    allDepthSorted = false
                }
            };
        }
        
        [BurstCompile]
        public static FrustumPlanes.IntersectResult Intersect(NativeArray<Plane> cullingPlanes, ref AABB a)
        {
            float3 m = a.Center;
            float3 extent = a.Extents;

            var inCount = 0;
            for (int i = 0; i < cullingPlanes.Length; i++)
            {
                float3 normal = cullingPlanes[i].normal;
                float dist = math.dot(normal, m) + cullingPlanes[i].distance;
                float radius = math.dot(extent, math.abs(normal));
                if (dist + radius <= 0)
                    return FrustumPlanes.IntersectResult.Out;

                if (dist > radius)
                    inCount++;
            }

            return (inCount == cullingPlanes.Length) ? FrustumPlanes.IntersectResult.In : FrustumPlanes.IntersectResult.Partial;
        }
        
        [BurstCompile]
        private static AABB Transform(ref float4 packed1, ref float4 packed2, ref float4 packed3, ref AABB localBounds)
        {
            AABB transformed;
            float3 c0 = packed1.xyz;
            float3 c1 = new float3(packed1.w, packed2.xy);
            float3 c2 = new float3(packed2.zw, packed3.x);
            float3 c3 = packed3.yzw;
            transformed.Extents = math.abs(c0 * localBounds.Extents.x) + math.abs(c1 * localBounds.Extents.y) + math.abs(c2 * localBounds.Extents.z);
            var b = localBounds.Center;
            transformed.Center = c0 * b.x + c1 * b.y + c2 * b.z + c3;
            return transformed;
        }
    }
}