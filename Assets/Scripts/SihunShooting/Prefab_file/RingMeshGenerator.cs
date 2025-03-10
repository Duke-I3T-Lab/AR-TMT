using UnityEngine;

[RequireComponent(typeof(MeshFilter))]
public class RingMeshGenerator : MonoBehaviour
{
    [Header("Auto Fit Settings")]
    [Tooltip("Automatically read the parent's bounds to set the ring size.")]
    public bool autoFitToTarget = true;
    [Tooltip("Additional margin added around the target.")]
    public float additionalMargin = 0.1f;
    [Tooltip("Extra thickness for the outline ring.")]
    public float ringThickness = 0.1f;

    [Header("Fallback Ring Settings")]
    [Tooltip("Inner radius if auto-fit is disabled.")]
    public float innerRadius = 0.5f;
    [Tooltip("Outer radius if auto-fit is disabled.")]
    public float outerRadius = 1.0f;

    [Header("Mesh Settings")]
    [Tooltip("Number of segments around the ring.")]
    public int segments = 36;

    void Start()
    {
        if(autoFitToTarget)
        {
            FitToParent();
        }
        GenerateRing();
    }

    /// <summary>
    /// Reads the parent's bounds (via Renderer or Collider) and sets inner and outer radii.
    /// </summary>
    void FitToParent()
    {
        // Try to use a Renderer first.
        Renderer parentRenderer = GetComponentInParent<Renderer>();
        if (parentRenderer != null)
        {
            Bounds bounds = parentRenderer.bounds;
            // Calculate an effective "radius" using the maximum of width and height.
            float targetRadius = Mathf.Max(bounds.size.x, bounds.size.y) * 0.5f;
            innerRadius = targetRadius + additionalMargin;
            outerRadius = innerRadius + ringThickness;
            return;
        }

        // If no Renderer is found, try a Collider.
        Collider parentCollider = GetComponentInParent<Collider>();
        if (parentCollider != null)
        {
            Bounds bounds = parentCollider.bounds;
            float targetRadius = Mathf.Max(bounds.size.x, bounds.size.y) * 0.5f;
            innerRadius = targetRadius + additionalMargin;
            outerRadius = innerRadius + ringThickness;
        }
    }

    /// <summary>
    /// Generates a flat ring mesh in the XY plane.
    /// </summary>
    void GenerateRing()
    {
        Mesh mesh = new Mesh();
        Vector3[] vertices = new Vector3[(segments + 1) * 2];
        int[] triangles = new int[segments * 6];

        // Angle between segments (in radians)
        float deltaAngle = 2.0f * Mathf.PI / segments;
        for (int i = 0; i <= segments; i++)
        {
            float angle = i * deltaAngle;
            float cos = Mathf.Cos(angle);
            float sin = Mathf.Sin(angle);

            // Outer vertex (first in pair)
            vertices[i * 2] = new Vector3(outerRadius * cos, outerRadius * sin, 0);
            // Inner vertex (second in pair)
            vertices[i * 2 + 1] = new Vector3(innerRadius * cos, innerRadius * sin, 0);
        }

        int triIndex = 0;
        for (int i = 0; i < segments; i++)
        {
            int index0 = i * 2;
            int index1 = i * 2 + 1;
            int index2 = (i + 1) * 2;
            int index3 = (i + 1) * 2 + 1;

            // First triangle
            triangles[triIndex++] = index0;
            triangles[triIndex++] = index1;
            triangles[triIndex++] = index2;
            // Second triangle
            triangles[triIndex++] = index2;
            triangles[triIndex++] = index1;
            triangles[triIndex++] = index3;
        }

        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();

        MeshFilter filter = GetComponent<MeshFilter>();
        filter.mesh = mesh;
    }
}
