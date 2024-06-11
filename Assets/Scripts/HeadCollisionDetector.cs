using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HeadCollisionDetector : MonoBehaviour
{
    public KnightofWood agent;
    // Start is called before the first frame update
    void Start()
    {
        if (agent == null)
        {
            Debug.LogError("KnightofWood reference not set in the Inspector.");
        }
    }

    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("ground"))
        {
            Debug.Log("Head collided with: " + collision.gameObject.name);
            agent.AddReward(-0.1f); // Optional: Set a negative reward
        }
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.CompareTag("Wall"))
        {
            Debug.Log("Head triggered with: " + other.gameObject.name);
            agent.SetReward(-1f); // Optional: Set a positive reward
            agent.EndEpisode();
        }
    }
}
