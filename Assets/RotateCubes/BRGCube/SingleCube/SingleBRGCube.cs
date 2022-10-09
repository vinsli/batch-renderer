using System;
using RotateCubes.BRGCube;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using UnityEngine;
using UnityEngine.Rendering;

public unsafe class SingleBRGCube : MonoBehaviour
{
    private const int kSizeOfMatrix = sizeof(float) * 4 * 4;
    private const int kSizeOfPackedMatrix = sizeof(float) * 4 * 3;
    private const int kSizeOfFloat4 = sizeof(float) * 4;
    private const int kBytesPerInstance = (kSizeOfPackedMatrix * 2) + kSizeOfFloat4;
    private const int kExtraBytes = kSizeOfMatrix * 2;
    private const int kNumInstances = 1;
    
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
    
    int BufferCountForInstances(int bytesPerInstance, int numInstances, int extraBytes = 0)
    {
        bytesPerInstance = (bytesPerInstance + sizeof(int) - 1) / sizeof(int) * sizeof(int);
        extraBytes = (extraBytes + sizeof(int) - 1) / sizeof(int) * sizeof(int);
        int totalBytes = bytesPerInstance * numInstances + extraBytes;
        return totalBytes / sizeof(int);
    }

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

        int bufferCount = BufferCountForInstances(kBytesPerInstance, kNumInstances, kExtraBytes);
        m_CopySrc = new GraphicsBuffer(target,
            bufferCount,
            sizeof(int));
        m_InstanceData = new GraphicsBuffer(target,
            BufferSize(bufferCount) / sizeof(int),
            sizeof(int));

        // Create a constant buffer for BRG global values
        m_Globals = new GraphicsBuffer(GraphicsBuffer.Target.Constant,
            1,
            UnsafeUtility.SizeOf<BatchRendererGroupGlobals>());
        m_Globals.SetData(new [] { BatchRendererGroupGlobals.Default });
        
        var zero = new Matrix4x4[1] { Matrix4x4.zero };

        var matrices = new Matrix4x4[kNumInstances]
        {
            Matrix4x4.Translate(new Vector3(0, 0, 0))
        };
        
        var objectToWorld = new PackedMatrix[kNumInstances]
        {
            new PackedMatrix(matrices[0])
        };
        
        var worldToObject = new PackedMatrix[kNumInstances]
        {
            new PackedMatrix(matrices[0].inverse)
        };
        
        var colors = new Vector4[kNumInstances]
        {
            new Vector4(1, 0, 0, 1)
        };

        uint byteAddressObjectToWorld = kSizeOfPackedMatrix * 2;
        uint byteAddressWorldToObject = byteAddressObjectToWorld + kSizeOfPackedMatrix * kNumInstances;
        uint byteAddressColor = byteAddressWorldToObject + kSizeOfPackedMatrix * kNumInstances;

        // Upload our instance data to the GraphicsBuffer, from where the shader can load them.
        m_CopySrc.SetData(zero, 0, 0, 1);
        m_CopySrc.SetData(objectToWorld, 0, (int)((byteAddressObjectToWorld + 0) / kSizeOfPackedMatrix), objectToWorld.Length);
        m_CopySrc.SetData(worldToObject, 0, (int)((byteAddressWorldToObject + 0)  / kSizeOfPackedMatrix), worldToObject.Length);
        m_CopySrc.SetData(colors, 0, (int)((byteAddressColor + 0)  / kSizeOfFloat4), colors.Length);

        int dstSize = m_CopySrc.count * m_CopySrc.stride;
        memcpy.SetBuffer(0, "src", m_CopySrc);
        memcpy.SetBuffer(0, "dst", m_InstanceData);
        memcpy.SetInt("dstOffset", BufferOffset);
        memcpy.SetInt("dstSize", dstSize);
        memcpy.Dispatch(0, dstSize / (64 * 4) + 1, 1, 1);
        
        var metadata = new NativeArray<MetadataValue>(3, Allocator.Temp);
        metadata[0] = new MetadataValue { NameID = Shader.PropertyToID("unity_ObjectToWorld"), Value = 0x80000000 | byteAddressObjectToWorld, };
        metadata[1] = new MetadataValue { NameID = Shader.PropertyToID("unity_WorldToObject"), Value = 0x80000000 | byteAddressWorldToObject, };
        metadata[2] = new MetadataValue { NameID = Shader.PropertyToID("_BaseColor"), Value = 0x80000000 | byteAddressColor, };
        
        m_BatchID = m_BRG.AddBatch(metadata, m_InstanceData.bufferHandle, (uint)BufferOffset, (uint)BufferWindowSize);
    }

    private void Update()
    {
        Shader.SetGlobalConstantBuffer(BatchRendererGroupGlobals.kGlobalsPropertyId, m_Globals, 0, m_Globals.stride);
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
        
        var drawCommands = (BatchCullingOutputDrawCommands*)cullingOutput.drawCommands.GetUnsafePtr();
        
        drawCommands->drawCommands = (BatchDrawCommand*)UnsafeUtility.Malloc(UnsafeUtility.SizeOf<BatchDrawCommand>(), alignment, Allocator.TempJob);
        drawCommands->drawRanges = (BatchDrawRange*)UnsafeUtility.Malloc(UnsafeUtility.SizeOf<BatchDrawRange>(), alignment, Allocator.TempJob);
        drawCommands->visibleInstances = (int*)UnsafeUtility.Malloc(kNumInstances * sizeof(int), alignment, Allocator.TempJob);
        drawCommands->drawCommandPickingInstanceIDs = null;

        drawCommands->drawCommandCount = 1;
        drawCommands->drawRangeCount = 1;
        drawCommands->visibleInstanceCount = kNumInstances;

        drawCommands->instanceSortingPositions = null;
        drawCommands->instanceSortingPositionFloatCount = 0;
        
        drawCommands->drawCommands[0].visibleOffset = 0;
        drawCommands->drawCommands[0].visibleCount = kNumInstances;
        drawCommands->drawCommands[0].batchID = m_BatchID;
        drawCommands->drawCommands[0].materialID = m_MaterialID;
        drawCommands->drawCommands[0].meshID = m_MeshID;
        drawCommands->drawCommands[0].submeshIndex = 0;
        drawCommands->drawCommands[0].splitVisibilityMask = 0xff;
        drawCommands->drawCommands[0].flags = 0;
        drawCommands->drawCommands[0].sortingPosition = 0;
        
        drawCommands->drawRanges[0].drawCommandsBegin = 0;
        drawCommands->drawRanges[0].drawCommandsCount = 1;
        
        drawCommands->drawRanges[0].filterSettings = new BatchFilterSettings { renderingLayerMask = 0xffffffff, };
        
        for (int i = 0; i < kNumInstances; ++i)
            drawCommands->visibleInstances[i] = i;
        
        return new JobHandle();
    }
}
