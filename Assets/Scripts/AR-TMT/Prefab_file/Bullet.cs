using UnityEngine;

public class Bullet : MonoBehaviour
{
    public float speed = 50f;          // Speed of the bullet
    public float maxLifetime = 5f;     // Lifetime before the bullet is destroyed
    public GameObject impactEffect;    // Optional: Effect to play on impact

    private void Start()
    {
        // Destroy the bullet automatically after maxLifetime
        Destroy(gameObject, maxLifetime);
    }

    private void Update()
    {
        // Move the bullet forward every frame
        transform.Translate(Vector3.forward * speed * Time.deltaTime);
    }

    private void OnTriggerEnter(Collider other)
    {
        // Check for collisions with targets or other objects
        Debug.Log($"Bullet hit: {other.gameObject.name}");

        // Optional: Instantiate an impact effect at the collision point
        if (impactEffect != null)
        {
            Instantiate(impactEffect, transform.position, transform.rotation);
        }

        // Destroy the bullet on impact
        Destroy(gameObject);
    }
}