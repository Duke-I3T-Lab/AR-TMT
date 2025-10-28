using System.Collections;
using UnityEngine;
using UnityEngine.XR.MagicLeap;

public class GlobalDimmer : MonoBehaviour
{
    public float fadeValue = 1.0f;
    void Start()
    {
        MLGlobalDimmer.SetValue(fadeValue);
    }
}