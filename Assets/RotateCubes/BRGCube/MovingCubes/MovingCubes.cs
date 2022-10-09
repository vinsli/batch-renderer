using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using Random = Unity.Mathematics.Random;
using static RotateCubes.BRGCube.BRGCubeUtility;

namespace RotateCubes.BRGCube.MovingCubes
{
    public class MovingCubes : MonoBehaviour
    {
        public int instances;
        public float rotateSpeed;

        public Mesh mesh;
        public Material material;
        public ComputeShader memcpy;

        private BatchRendererGroup m_BRG;

        private GraphicsBuffer m_InstanceData;
        private GraphicsBuffer m_CopySrc;
        private GraphicsBuffer m_Globals;
        private BatchID m_BatchID;
        private BatchMeshID m_MeshID;
        private BatchMaterialID m_MaterialID;

        private bool UseConstantBuffer => BatchRendererGroup.BufferTarget == BatchBufferTarget.ConstantBuffer;

        private int BufferSize(int bufferCount) => bufferCount * sizeof(int);
        private int BufferOffset => 0;
        private int BufferWindowSize => UseConstantBuffer ? SystemInfo.maxConstantBufferSize : 0;

        private uint _byteAddressObjectToWorld = (uint)kSizeOfPackedMatrix * 2;
        private uint _byteAddressWorldToObject;
        private uint _byteAddressColor;

        private Matrix4x4[] _zero = {Matrix4x4.zero};
        private PackedMatrix[] _objectToWorld;
        private PackedMatrix[] _worldToObject;
        private float4[] _colors;
        private float3[] _positions;
        private quaternion[] _rotations;

        private float _step;
        private Random _random;

        void Start()
        {
            // Create the BatchRendererGroup and register assets
            m_BRG = new BatchRendererGroup(this.OnPerformCulling, IntPtr.Zero);
            m_MeshID = m_BRG.RegisterMesh(mesh);
            m_MaterialID = m_BRG.RegisterMaterial(material);

            // Create the buffer that holds our instance data
            var target = GraphicsBuffer.Target.Raw;
            if (SystemInfo.graphicsDeviceType is GraphicsDeviceType.OpenGLCore or GraphicsDeviceType.OpenGLES3)
                target |= GraphicsBuffer.Target.Constant;

            int bufferCount = BufferCountForInstances(kBytesPerInstance, (uint)instances, kExtraBytes);
            m_CopySrc = new GraphicsBuffer(target, bufferCount, sizeof(int));
            m_InstanceData = new GraphicsBuffer(target, BufferSize(bufferCount) / sizeof(int), sizeof(int));

            // Create a constant buffer for BRG global values
            m_Globals = new GraphicsBuffer(GraphicsBuffer.Target.Constant, 1, UnsafeUtility.SizeOf<BatchRendererGroupGlobals>());
            m_Globals.SetData(new[] {BatchRendererGroupGlobals.Default});

            _byteAddressWorldToObject = (uint)(_byteAddressObjectToWorld + kSizeOfPackedMatrix * instances);
            _byteAddressColor = (uint)(_byteAddressWorldToObject + kSizeOfPackedMatrix * instances);

            var metadata = new NativeArray<MetadataValue>(3, Allocator.Temp);
            metadata[0] = new MetadataValue {NameID = Shader.PropertyToID("unity_ObjectToWorld"), Value = 0x80000000 | _byteAddressObjectToWorld,};
            metadata[1] = new MetadataValue {NameID = Shader.PropertyToID("unity_WorldToObject"), Value = 0x80000000 | _byteAddressWorldToObject,};
            metadata[2] = new MetadataValue {NameID = Shader.PropertyToID("_BaseColor"), Value = 0x80000000 | _byteAddressColor,};

            m_BatchID = m_BRG.AddBatch(metadata, m_InstanceData.bufferHandle, (uint) BufferOffset, (uint) BufferWindowSize);
            
            _objectToWorld = new PackedMatrix[instances];
            _worldToObject = new PackedMatrix[instances];
            
            _colors = new float4[instances];
            _positions = new float3[instances];
            _rotations = new quaternion[instances];
            
            _random = new Random(83729);
            InitializeCubes();
        }
        
        private void InitializeCubes()
        {
            for (int i = 0; i < instances; i++)
            {
                _colors[i] = new float4(_random.NextFloat3(), 1);
                var randPos = _random.NextFloat3Direction() * math.pow(_random.NextFloat(0, 1), 1f / 3f);
                _positions[i] = randPos * 10;
                _rotations[i] = _random.NextQuaternionRotation();
            }
        }

        private void UpdateCubeData()
        {
            for (int i = 0; i < instances; i++)
            {
                var rot = _rotations[i] * Quaternion.AngleAxis(Time.deltaTime * rotateSpeed, Vector3.up);
                _rotations[i] = rot;
                var cubeMatrix = Matrix4x4.TRS(_positions[i], rot, Vector3.one);
                _objectToWorld[i] = new PackedMatrix(cubeMatrix);
                _worldToObject[i] = new PackedMatrix(cubeMatrix.inverse);
            }
        }

        private void Update()
        {
            Shader.SetGlobalConstantBuffer(BatchRendererGroupGlobals.kGlobalsPropertyId, m_Globals, 0, m_Globals.stride);
            UpdateCubeData();
            UploadCubeData();
        }

        private void UploadCubeData()
        {
            // Upload our instance data to the GraphicsBuffer, from where the shader can load them.
            m_CopySrc.SetData(_zero, 0, 0, 1);
            m_CopySrc.SetData(_objectToWorld, 0, (int) ((_byteAddressObjectToWorld + 0) / kSizeOfPackedMatrix), _objectToWorld.Length);
            m_CopySrc.SetData(_worldToObject, 0, (int) ((_byteAddressWorldToObject + 0) / kSizeOfPackedMatrix), _worldToObject.Length);
            m_CopySrc.SetData(_colors, 0, (int) ((_byteAddressColor + 0) / kSizeOfFloat4), _colors.Length);

            int dstSize = m_CopySrc.count * m_CopySrc.stride;
            memcpy.SetBuffer(0, "src", m_CopySrc);
            memcpy.SetBuffer(0, "dst", m_InstanceData);
            memcpy.SetInt("dstOffset", BufferOffset);
            memcpy.SetInt("dstSize", dstSize);
            memcpy.Dispatch(0, dstSize / (64 * 4) + 1, 1, 1);
        }

        private void OnDisable()
        {
            m_CopySrc.Dispose();
            m_InstanceData.Dispose();
            m_BRG.Dispose();
            m_Globals.Dispose();
        }

        public unsafe JobHandle OnPerformCulling(
            BatchRendererGroup rendererGroup,
            BatchCullingContext cullingContext,
            BatchCullingOutput cullingOutput,
            IntPtr userContext)
        {
            int alignment = UnsafeUtility.AlignOf<long>();

            var drawCommands = (BatchCullingOutputDrawCommands*) cullingOutput.drawCommands.GetUnsafePtr();

            drawCommands->drawCommands = (BatchDrawCommand*) UnsafeUtility.Malloc(UnsafeUtility.SizeOf<BatchDrawCommand>(), alignment, Allocator.TempJob);
            drawCommands->drawRanges = (BatchDrawRange*) UnsafeUtility.Malloc(UnsafeUtility.SizeOf<BatchDrawRange>(), alignment, Allocator.TempJob);
            drawCommands->visibleInstances = (int*) UnsafeUtility.Malloc(instances * sizeof(int), alignment, Allocator.TempJob);
            drawCommands->drawCommandPickingInstanceIDs = null;

            drawCommands->drawCommandCount = 1;
            drawCommands->drawRangeCount = 1;
            drawCommands->visibleInstanceCount = instances;

            drawCommands->instanceSortingPositions = null;
            drawCommands->instanceSortingPositionFloatCount = 0;

            drawCommands->drawCommands[0].visibleOffset = 0;
            drawCommands->drawCommands[0].visibleCount = (uint)instances;
            drawCommands->drawCommands[0].batchID = m_BatchID;
            drawCommands->drawCommands[0].materialID = m_MaterialID;
            drawCommands->drawCommands[0].meshID = m_MeshID;
            drawCommands->drawCommands[0].submeshIndex = 0;
            drawCommands->drawCommands[0].splitVisibilityMask = 0xff;
            drawCommands->drawCommands[0].flags = 0;
            drawCommands->drawCommands[0].sortingPosition = 0;

            drawCommands->drawRanges[0].drawCommandsBegin = 0;
            drawCommands->drawRanges[0].drawCommandsCount = 1;  

            drawCommands->drawRanges[0].filterSettings = new BatchFilterSettings {renderingLayerMask = 0xffffffff,};

            for (int i = 0; i < instances; ++i)
                drawCommands->visibleInstances[i] = i;

            return new JobHandle();
        }
    }
}