using UnityEngine;

public class SpinningEffect : MonoBehaviour
{
    public float spinSpeed = 90f; // Degrees per second
    public enum SpinAxis { X, Y, Z }
    public SpinAxis spinAxis = SpinAxis.Y;

    private Vector3 rotationAxis;

    void Start()
    {
        // Set the rotation axis based on the enum
        switch (spinAxis)
        {
            case SpinAxis.X:
                rotationAxis = Vector3.right;
                break;
            case SpinAxis.Y:
                rotationAxis = Vector3.up;
                break;
            case SpinAxis.Z:
                rotationAxis = Vector3.forward;
                break;
        }

        // Optional: Add slight randomness to speed or axis
        spinSpeed *= Random.Range(0.8f, 1.2f);
    }

    void Update()
    {
        transform.Rotate(rotationAxis * spinSpeed * Time.deltaTime);
    }
}
