using UnityEngine;

public class GenerateIrregular3DClutter : MonoBehaviour
{
    public int minVertices = 5; // Minimum number of polygon edges
    public int maxVertices = 10; // Maximum number of polygon edges
    public float minExtrudeDepth = 0.05f; // Minimum depth for 3D effect
    public float maxExtrudeDepth = 0.2f; // Maximum depth for more variation
    public float noiseStrength = 0.05f; // Strength of the shape irregularity
    public float minScale = 0.1f; // Minimum scale
    public float maxScale = 0.3f; // Maximum scale

    private MeshFilter meshFilter;
    private MeshRenderer meshRenderer;

    void Start()
    {
        GenerateClutterShape();
    }

    void GenerateClutterShape()
    {
        // ✅ Ensure the GameObject has MeshFilter and MeshRenderer
        meshFilter = GetComponent<MeshFilter>() ?? gameObject.AddComponent<MeshFilter>();
        meshRenderer = GetComponent<MeshRenderer>() ?? gameObject.AddComponent<MeshRenderer>();

        Mesh mesh = new Mesh();
        mesh.name = "Generated 3D Irregular Clutter";

        int vertexCount = Random.Range(minVertices, maxVertices + 1);
        float extrudeDepth = Random.Range(minExtrudeDepth, maxExtrudeDepth); // Random extrusion depth

        Vector3[] vertices = new Vector3[vertexCount * 2]; // Front & back faces
        int[] triangles = new int[(vertexCount - 2) * 6 + vertexCount * 6]; // Front, back, and side faces

        // ✅ Generate random irregular polygon shape (front face)
        for (int i = 0; i < vertexCount; i++)
        {
            float angle = (i / (float)vertexCount) * Mathf.PI * 2;
            float radius = Random.Range(0.1f, 0.3f); // Adjust radius for better variety

            // ✅ Add Perlin noise to make the shape more irregular
            float noise = Mathf.PerlinNoise(angle * 2, Time.time) * noiseStrength;
            radius += noise; 

            vertices[i] = new Vector3(Mathf.Cos(angle) * radius, Mathf.Sin(angle) * radius, 0); // Front face
            vertices[i + vertexCount] = vertices[i] + new Vector3(0, 0, -extrudeDepth); // Back face
        }

        // ✅ Create triangles for front and back faces
        int triIndex = 0;
        for (int i = 1; i < vertexCount - 1; i++)
        {
            triangles[triIndex++] = 0;
            triangles[triIndex++] = i;
            triangles[triIndex++] = i + 1;

            triangles[triIndex++] = vertexCount;
            triangles[triIndex++] = vertexCount + i + 1;
            triangles[triIndex++] = vertexCount + i;
        }

        // ✅ Create triangles for side faces (connecting front and back faces)
        for (int i = 0; i < vertexCount; i++)
        {
            int next = (i + 1) % vertexCount;

            triangles[triIndex++] = i;
            triangles[triIndex++] = next;
            triangles[triIndex++] = vertexCount + next;

            triangles[triIndex++] = i;
            triangles[triIndex++] = vertexCount + next;
            triangles[triIndex++] = vertexCount + i;
        }

        // ✅ Assign mesh data
        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        mesh.Optimize();
        meshFilter.mesh = mesh;

        // ✅ Apply a semi-transparent AR-friendly material
        ApplyClutterMaterial();

        // ✅ Apply random scaling for variety (clamped to prevent unrealistic sizes)
        float randomScale = Random.Range(minScale, maxScale);
        transform.localScale = new Vector3(randomScale, randomScale, randomScale);

        // ✅ Apply random rotation to further diversify clutter appearance
        transform.rotation = Random.rotation;

        // ✅ Enable shadows for realism
        // meshRenderer.receiveShadows = true;
        // meshRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.On;

        Debug.Log("Generated 3D irregular clutter object.");
    }

    void ApplyClutterMaterial()
    {
        Shader clutterShader = Shader.Find("Universal Render Pipeline/Lit");
        Material clutterMaterial = new Material(clutterShader);

        // ✅ Use dithered transparency to prevent stereo mismatch
        clutterMaterial.SetFloat("_AlphaClip", 1);
        clutterMaterial.SetFloat("_Surface", 1);
        clutterMaterial.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
        clutterMaterial.color = new Color(0.6f, 0.6f, 0.6f, 0.4f); // Semi-transparent

        meshRenderer.material = clutterMaterial;
    }
}
