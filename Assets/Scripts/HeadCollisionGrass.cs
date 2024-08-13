using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HeadCollisionGrass : MonoBehaviour
{
    public KnightofGrass agent;
    // Start is called before the first frame update
    void Start()
    {
        if (agent == null)
        {
            Debug.LogError("KnightofGrass reference not set in the Inspector.");
        }
    }

    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("ground"))
        {
            agent.AddReward(-1f); // Optional: Set a negative reward
        }
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.CompareTag("Wall"))
        {
            Debug.Log("Head triggered with: " + other.gameObject.name);
            agent.AddReward(-1f); 
            agent.EndEpisode();
        }
    }
}
