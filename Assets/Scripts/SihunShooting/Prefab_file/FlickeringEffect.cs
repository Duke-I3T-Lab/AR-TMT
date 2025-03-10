using UnityEngine;

public class FlickeringEffect : MonoBehaviour
{

    public float flickerSpeed = 2.0f; // Speed of transparency flicker
    private Renderer objectRenderer;
    private Color originalColor;
    private float phaseOffset; // Unique phase shift per object

    void Start()
    {
        objectRenderer = GetComponent<Renderer>();

        if (objectRenderer != null)
        {
            originalColor = objectRenderer.material.color;

            // Ensure material supports transparency
            objectRenderer.material.SetFloat("_Surface", 1); // 0 = Opaque, 1 = Transparent
            objectRenderer.material.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
            objectRenderer.material.EnableKeyword("_ALPHABLEND_ON");
        }

        // ✅ Add a random phase shift so each object moves independently
        phaseOffset = Random.Range(0f, Mathf.PI * 2);
    }
    void Update()
    {

        if (objectRenderer != null)
        {
            // Apply Transparency Flicker Effect (with Phase Offset)
            float alpha = Mathf.PingPong((Time.time + phaseOffset) * flickerSpeed, 1.0f);
            objectRenderer.material.color = new Color(originalColor.r, originalColor.g, originalColor.b, alpha);
        }
    }
}