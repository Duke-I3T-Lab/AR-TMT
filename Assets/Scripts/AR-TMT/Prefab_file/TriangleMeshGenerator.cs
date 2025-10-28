using UnityEngine;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class TrianglePrismGenerator : MonoBehaviour
{
    private MeshFilter meshFilter;
    private MeshRenderer meshRenderer;

    void Awake()
    {
        meshFilter = GetComponent<MeshFilter>();
        meshRenderer = GetComponent<MeshRenderer>();

        if (meshFilter == null)
            meshFilter = gameObject.AddComponent<MeshFilter>();

        if (meshRenderer == null)
            meshRenderer = gameObject.AddComponent<MeshRenderer>();
    }

    void Start()
    {
        GenerateTrianglePrism();
        transform.LookAt(Camera.main.transform.position); // Face the main camera

    }

    void GenerateTrianglePrism()
    {
        Mesh mesh = new Mesh();
        mesh.name = "Generated Triangular Prism";

        // Define 6 vertices (adjusted so the triangle faces forward)
        Vector3[] vertices = new Vector3[]
        {
            // Front face (Facing the camera, XY plane)
            new Vector3(-0.5f, 0, 0),  // Bottom left
            new Vector3(0.5f, 0, 0),   // Bottom right
            new Vector3(0, 1, 0),      // Top center

            // Back face (Depth offset)
            new Vector3(-0.5f, 0, -0.5f),  // Bottom left (back)
            new Vector3(0.5f, 0, -0.5f),   // Bottom right (back)
            new Vector3(0, 1, -0.5f)       // Top center (back)
        };

        // Define triangles (2 faces + 3 sides)
        int[] triangles = new int[]
        {
            // Front face (Clockwise order)
            0, 2, 1, 

            // Back face
            3, 4, 5,

            // Side Faces
            0, 1, 4,  0, 4, 3,  // Bottom side
            1, 2, 5,  1, 5, 4,  // Right side
            2, 0, 3,  2, 3, 5   // Left side
        };

        // Assign mesh properties
        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.RecalculateNormals();

        // ✅ Assign the generated mesh to the MeshFilter
        meshFilter.mesh = mesh;

        // Debugging
        Debug.Log("3D Triangular Prism Generated and Rotated Properly!");

        // Apply material if none is set
        if (meshRenderer.material == null)
        {
            meshRenderer.material = new Material(Shader.Find("Universal Render Pipeline/Lit"));
            meshRenderer.material.color = Color.red;
        }
    }
}
