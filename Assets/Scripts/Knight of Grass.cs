using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Unity.VisualScripting;

//this is a knight that has a sword and shield. Using its vision, its goal is to hit the opposing knight with the sword. However, if it hits the opposing knights shield or sword, this doesnt count as a hit. The knight can also block the opposing knights sword with its shied or sword. The knight can also move around the arena, as can the opposing knight. This knight starts across from but far from
//the opposing knight. The sword and shielf are attached to this knights lower right and lower left arms respectively. im using a camera for vision, so the knight can see the opposing knight we can find the opposing knights sword shield and body objects since this knight the other knigth and the floor are in a prefab
public class KnightofGrass : Agent {
    private Rigidbody rb;
    JointDriveController m_JdController;
    OrientationCubeController m_OrientationCube;

    public Transform Opponent; // Add this field
    private float previousDistanceToOpponent;

    public Transform Body;
    public Transform UpperArm_L;
    public Transform LowerArm_L;
    public Transform UpperArm_R;
    public Transform LowerArm_R;
    public Transform UpperLeg_L;
    public Transform LowerLeg_L;
    public Transform UpperLeg_R;
    public Transform LowerLeg_R;
    public Transform Head;

    public override void Initialize()
    {
        // Debug.Log("Initialize called");
        // Instantiate the OrientationCubeController directly
        GameObject orientationCubeObject = new GameObject("OrientationCube");
        m_OrientationCube = orientationCubeObject.AddComponent<OrientationCubeController>();

        // Move the orientation cube under the agent in the hierarchy
        orientationCubeObject.transform.parent = transform;
        
        GameObject jointdriveobject = new GameObject("JointDrive");
        m_JdController = jointdriveobject.AddComponent<JointDriveController>();
        m_JdController.maxJointForceLimit = 50f;
        m_JdController.maxJointSpring = 10f;
        m_JdController.jointDampen = 5f;

        m_JdController.transform.parent = transform;

        //Setup each body part
        SetupBodyPart(UpperArm_L);
        SetupBodyPart(LowerArm_L);
        SetupBodyPart(UpperArm_R);
        SetupBodyPart(LowerArm_R);
        SetupBodyPart(UpperLeg_L);
        SetupBodyPart(LowerLeg_L);
        SetupBodyPart(UpperLeg_R);
        SetupBodyPart(LowerLeg_R);
        SetupBodyPart(Head);
        SetupBodyPart(Body);
    }

    void SetupBodyPart(Transform bodyPart)
    {
        if (bodyPart != null)
        {
            m_JdController.SetupBodyPart(bodyPart);
        }
        else
        {
            Debug.LogWarning("Body part is null!");
        }
    }

    void Start() {
        rb = GetComponent<Rigidbody>();
        if (rb == null)
        {
            Debug.LogError("Rigidbody is not attached to the agent.");
        }
    }

    public override void OnEpisodeBegin() {
        transform.position = transform.position = new Vector3(0, 2.38f, 4);  // Adjust the y-value as needed for correct height above the floor
        transform.rotation = Quaternion.Euler(0, 0, 0);  // Adjust the Euler angles as needed to face the opponent
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        foreach (var bodyPart in m_JdController.bodyPartsList)
        {
            bodyPart.Reset(bodyPart);
        }
        if (Opponent != null)
        {
            previousDistanceToOpponent = Vector3.Distance(Body.transform.position, Opponent.position);
        }
    }

    private void CollectObservationBodyPart(BodyPart bp, VectorSensor sensor) {
        // Debug.Log("collected body part obs");
        
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.velocity));
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.angularVelocity));
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.position - Body.position));

        sensor.AddObservation(bp.rb.transform.localRotation);
        sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit);
    }

    public override void CollectObservations(VectorSensor sensor) {
        // Debug.Log("collected obs");
        var cubeForward = m_OrientationCube.transform.forward;

        //ragdoll's avg vel
        var avgVel = GetAvgVelocity();

        //current ragdoll velocity. normalized
        sensor.AddObservation(avgVel);
        //avg body vel relative to cube
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(avgVel));
        //rotation deltas
        sensor.AddObservation(Quaternion.FromToRotation(Body.forward, cubeForward));
        sensor.AddObservation(Quaternion.FromToRotation(Head.forward, cubeForward));

        foreach (var bodyPart in m_JdController.bodyPartsList)
        {
            CollectObservationBodyPart(bodyPart, sensor);
        } 

        // Debug.Log($"Total Observations: {sensor.ObservationSize()}");
    }

    //the knight can both move and rotate its upper and lower right and left arms
    //as well as its right and left foot and knee
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // Debug.Log("OnActionReceived called");
        var bpDict = m_JdController.bodyPartsDict;
        var i = 0;

        var continuousActions = actionBuffers.ContinuousActions;
        // Debug.Log($"ContinuousActions Length: {continuousActions.Length}");
        foreach (var bp in bpDict.Values)
        {
            if (continuousActions.Length >= (i * 3 + i) && bp != null && bp.joint != null)
            {
                bp.SetJointTargetRotation(continuousActions[i * 3], continuousActions[i * 3 + 1], continuousActions[i * 3 + 2]);
                bp.SetJointStrength(continuousActions[bpDict.Count * 3 + i]);
            }
        }
        if (Opponent != null)
        {
            float currentDistanceToOpponent = Vector3.Distance(Body.transform.position, Opponent.position);
            Debug.Log("Distance to opponent: " + currentDistanceToOpponent);
            if (currentDistanceToOpponent < previousDistanceToOpponent)
            {
                AddReward(0.01f); // Reward for getting closer
            }
            else if (currentDistanceToOpponent > previousDistanceToOpponent)
            {
                AddReward(-0.01f); // Reward for getting closer
            }
            previousDistanceToOpponent = currentDistanceToOpponent;
        }
    }

    private void OnCollisionEnter(Collision collision) {
        Debug.Log("OnCollisionEnter called");
        Debug.Log(collision.gameObject.name);
        if (collision.gameObject.name == "WallRight" || collision.gameObject.name == "WallLeft" || collision.gameObject.name == "WallWood" || collision.gameObject.name == "WallGrass") {
            Debug.Log("OnCollisionEnter executed with " + collision.gameObject.name);
            SetReward(-2f); // Out of the arena
            EndEpisode();
        }
    }

    private void OnTriggerEnter(Collider other)
    {  
        Debug.Log("OnTriggerEnter called with " + other.gameObject.name);
        if (other.gameObject.name == "WallRight" || other.gameObject.name == "WallLeft" || other.gameObject.name == "WallWood" || other.gameObject.name == "WallGrass") {
            Debug.Log("OnTriggerEnter executed with " + other.gameObject.name);
            SetReward(-2f); // Out of the arena
            EndEpisode();
        }
        if (other.TryGetComponent(out Wall wall)) {
            Debug.Log("OnTriggerEnter executed with " + other.gameObject.name);
            SetReward(-2f); // Out of the arena
            EndEpisode();
        }
        // if(other.TryGetComponent<SwordGrass>(out SwordGrass SwordGrass)) {
        //     SetReward(-1f); // Hit by opponent's sword
        //     EndEpisode();
        // }
    }

    public void OnBlockedHit() {
        AddReward(0.1f); // Hit the opposing shield or sword
    }

    public void OnSwordHitOpponent()
    {
        // Add a positive reward for hitting the opponent
        AddReward(1.0f);
        EndEpisode();
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        Debug.Log("Heuristic called");
        var continuousActions = actionsOut.ContinuousActions;

        // Set all actions to 0 initially
        for (int i = 0; i < continuousActions.Length; i++) {
            continuousActions[i] = 0f;
        }
        //a key moves head joint x, y, and z rotation
        //b key moves head joint negative x, y, and z rotation
        
        //.. continue this for upper lower left right arm, upper lower right and left leg, and head
        //then the same for strength values for each fo these
        //in total there hsould be 40
        // Rotation

        continuousActions[0] = Input.GetKey(KeyCode.A) ? 1.0f : 0.0f;
        continuousActions[1] = Input.GetKey(KeyCode.A) ? 1.0f : 0.0f;
        continuousActions[2] = Input.GetKey(KeyCode.A) ? 1.0f : 0.0f;
        continuousActions[3] = Input.GetKey(KeyCode.B) ? 1.0f : 0.0f;
        continuousActions[4] = Input.GetKey(KeyCode.B) ? 1.0f : 0.0f;
        continuousActions[5] = Input.GetKey(KeyCode.B) ? 1.0f : 0.0f;
        continuousActions[6] = Input.GetKey(KeyCode.C) ? 1.0f : 0.0f;
        continuousActions[7] = Input.GetKey(KeyCode.C) ? 1.0f : 0.0f;
        continuousActions[8] = Input.GetKey(KeyCode.C) ? 1.0f : 0.0f;
        continuousActions[9] = Input.GetKey(KeyCode.D) ? 1.0f : 0.0f;
        continuousActions[10] = Input.GetKey(KeyCode.D) ? 1.0f : 0.0f;
        continuousActions[11] = Input.GetKey(KeyCode.D) ? 1.0f : 0.0f;
        continuousActions[12] = Input.GetKey(KeyCode.E) ? 1.0f : 0.0f;
        continuousActions[13] = Input.GetKey(KeyCode.E) ? 1.0f : 0.0f;
        continuousActions[14] = Input.GetKey(KeyCode.E) ? 1.0f : 0.0f;
        continuousActions[15] = Input.GetKey(KeyCode.F) ? 1.0f : 0.0f;
        continuousActions[16] = Input.GetKey(KeyCode.F) ? 1.0f : 0.0f;
        continuousActions[17] = Input.GetKey(KeyCode.F) ? 1.0f : 0.0f;
        continuousActions[18] = Input.GetKey(KeyCode.G) ? 1.0f : 0.0f;
        continuousActions[19] = Input.GetKey(KeyCode.G) ? 1.0f : 0.0f;
        continuousActions[20] = Input.GetKey(KeyCode.G) ? 1.0f : 0.0f;
        continuousActions[21] = Input.GetKey(KeyCode.H) ? 1.0f : 0.0f;
        continuousActions[22] = Input.GetKey(KeyCode.H) ? 1.0f : 0.0f; 
        continuousActions[23] = Input.GetKey(KeyCode.H) ? 1.0f : 0.0f; 
        continuousActions[24] = Input.GetKey(KeyCode.I) ? 1.0f : 0.0f; 
        continuousActions[25] = Input.GetKey(KeyCode.I) ? 1.0f : 0.0f; 
        continuousActions[26] = Input.GetKey(KeyCode.I) ? 1.0f : 0.0f; 

        //stregth
        continuousActions[27] = Input.GetKey(KeyCode.J) ? 1.0f : 0.0f; 
        continuousActions[28] = Input.GetKey(KeyCode.K) ? 1.0f : 0.0f; 
        continuousActions[29] = Input.GetKey(KeyCode.L) ? 1.0f : 0.0f; 
        continuousActions[30] = Input.GetKey(KeyCode.M) ? 1.0f : 0.0f;
        continuousActions[31] = Input.GetKey(KeyCode.N) ? 1.0f : 0.0f;
        continuousActions[32] = Input.GetKey(KeyCode.O) ? 1.0f : 0.0f;
        continuousActions[33] = Input.GetKey(KeyCode.P) ? 1.0f : 0.0f;
        continuousActions[34] = Input.GetKey(KeyCode.Q) ? 1.0f : 0.0f;
        continuousActions[35] = Input.GetKey(KeyCode.R) ? 1.0f : 0.0f;

        Debug.Log($"Heuristic actions: {string.Join(", ", continuousActions)}");
    }

    Vector3 GetAvgVelocity()
    {
        Vector3 velSum = Vector3.zero;

        //ALL RBS
        int numOfRb = 0;
        foreach (var item in m_JdController.bodyPartsList)
        {
            numOfRb++;
            velSum += item.rb.velocity;
        }

        var avgVel = velSum / numOfRb;
        return avgVel;
    }

}

