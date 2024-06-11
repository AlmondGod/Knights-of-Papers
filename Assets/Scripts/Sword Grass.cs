using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SwordGrass : MonoBehaviour
{
    //we want to have this sword always be held by the 'Robot' model's child 'LowerArm_L.LowerArm
    //so we need to have a reference to the 'LowerArm_L.LowerArm' GameObject
    [SerializeField] private GameObject attachedArm;
    private Vector3 initialLocalPosition;
    private Quaternion initialLocalRotation;
    public KnightofGrass agent;

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

    private void OnCollisionEnter(Collision collision) {
        // Check if the sword hit the opponent
        if (collision.gameObject.name == "Knight of Wood")
        {
            Debug.Log("Hit the opponent!");
            // Call a method on the agent to add a reward for hitting the opponent
            agent.OnSwordHitOpponent();
        }

        if (collision.gameObject.name == "SwordW" || collision.gameObject.name == "ShieldW") {
            Debug.Log("Hit the opposing sword/shield!");
            // Call a method on the agent to add a reward for hitting the opponent
            agent.OnBlockedHit();
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        Debug.Log("Sword triggered with: " + other.gameObject.name);
        if (other.gameObject.CompareTag("whead"))
        {
            Debug.Log("Sword triggered with: " + other.gameObject.name);
            agent.SetReward(5f);
            agent.EndEpisode();
        }   
    }
}
