using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.XR.Interaction.Toolkit;
using UnityEngine.XR.Interaction.Toolkit.AffordanceSystem.Receiver.Rendering;

public class ButtonControl : MonoBehaviour
{
    
    // Start is called before the first frame update
    public Material defaultMaterial;
    public Material litMaterial;
    public int order = 0;

    private Color color;

    public UnityEvent<int> OnSelectEvent;

    void Start()
    {
        Gradient colorGradient = transform.parent.GetComponent<ChoicesControl>().gradient;
        color = colorGradient.Evaluate((float)order / (float)transform.parent.childCount);
        GetComponent<Renderer>().material = defaultMaterial;
        GetComponent<Renderer>().material.color = color;
    }

    // Update is called once per frame
    public void OnHoverEntered(HoverEnterEventArgs args)
    {
        GetComponent<Renderer>().material = litMaterial;
    }

    public void OnHoverExited(HoverExitEventArgs args)
    {
        GetComponent<Renderer>().material = defaultMaterial;
        GetComponent<Renderer>().material.color = color;
    }

    public void OnSelectEntering(SelectEnterEventArgs args)
    {
        Debug.Log("onselectenteddring");
        OnSelectEvent.Invoke(order);
    }
}
