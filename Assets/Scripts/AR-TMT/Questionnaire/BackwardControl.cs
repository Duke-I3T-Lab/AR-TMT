using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.XR.Interaction.Toolkit;

public class BackwardControl : MonoBehaviour
{
    // Start is called before the first frame update
    public Material defaultMaterial;
    public Material litMaterial;

    public UnityEvent OnSelectEvent;

    public void OnHoverEntered(HoverEnterEventArgs args)
    {
        GetComponent<Renderer>().material = litMaterial;
    }

    public void OnHoverExited(HoverExitEventArgs args)
    {
        GetComponent<Renderer>().material = defaultMaterial;
    }

    public void OnSelectEntering(SelectEnterEventArgs args)
    {
        OnSelectEvent.Invoke();
    }
}
