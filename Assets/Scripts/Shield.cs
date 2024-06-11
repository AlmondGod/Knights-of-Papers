using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Shield : MonoBehaviour
{
    [SerializeField] private GameObject attachedArm;
    private Vector3 initialLocalPosition;
    private Quaternion initialLocalRotation;

    // Start is called before the first frame update
    void Start()
    {
        transform.SetParent(attachedArm.transform);

        // Store the initial local position and rotation
        initialLocalPosition = transform.localPosition;
        initialLocalRotation = transform.localRotation;

        // Ensure the sword maintains the relative difference in position and rotation
        transform.localPosition = initialLocalPosition;
        transform.localRotation = initialLocalRotation;
    }

    // Update is called once per frame
    void Update()
    {
        // Ensure the sword maintains the relative difference in position and rotation
        transform.localPosition = initialLocalPosition;
        transform.localRotation = initialLocalRotation;
    }
}
