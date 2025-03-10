using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TextinPlaneDetection : MonoBehaviour
{
    void Start()
    {
        // 🔹 Set a reasonable scale
        Canvas myCanvas = GetComponent<Canvas>();
        myCanvas.transform.localScale = new Vector3(0.01f, 0.01f, 0.01f);

        // 🔹 Ensure the canvas faces the camera
        myCanvas.transform.LookAt(Camera.main.transform);
        myCanvas.transform.Rotate(0, 180, 0); // Flip to face user properly
    }

        void Update()
    {
        transform.LookAt(Camera.main.transform);
        transform.Rotate(0, 180, 0); // Flip to face the user correctly
    }
}
