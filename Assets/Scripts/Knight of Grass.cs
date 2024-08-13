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
    public Transform Opponent; 
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
        // Instantiate the OrientationCubeController directly
        GameObject orientationCubeObject = new GameObject("OrientationCube");
        m_OrientationCube = orientationCubeObject.AddComponent<OrientationCubeController>();

        // Move the orientation cube under the agent in the hierarchy
        orientationCubeObject.transform.parent = transform;
        
        GameObject jointdriveobject = new GameObject("JointDrive");
        m_JdController = jointdriveobject.AddComponent<JointDriveController>();
        m_JdController.maxJointForceLimit = 1000000000000000f;
        m_JdController.maxJointSpring = 100000000000000f;
        m_JdController.jointDampen = 0f;

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
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.velocity));
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.angularVelocity));
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.position - Body.position));

        sensor.AddObservation(bp.rb.transform.localRotation);
        sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit);
    }

    public override void CollectObservations(VectorSensor sensor) {
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

        //add a reward if the position of the head is above the position of the body
        if (Head.position.y > Body.position.y)
        {
            AddReward(0.1f);
        }
    }

    //the knight can both move and rotate its upper and lower right and left arms
    //as well as its right and left foot and knee
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var bpDict = m_JdController.bodyPartsDict;
        var continuousActions = actionBuffers.ContinuousActions;
        // Debug.Log(actionBuffers.ContinuousActions);

        foreach (var transform in bpDict.Keys)
        {
            //if bp is head dont rotate on the z axis
            //if bp is lower arm or lower leg dont rotate on z or y axis
            //if bp is upper leg dont rotate on y axis
            var bp = bpDict[transform];
            // Debug.Log("bp: " + bp);
            // Debug.log(continuousActions[0] + " " + continuousActions[1]);
            // Debug.Log("all continuous actions: " + continuousActions[0] + " " + continuousActions[1] + " " + continuousActions[2] + " " + continuousActions[3] + " " + continuousActions[4] + " " + continuousActions[5] + " " + continuousActions[6] + " " + continuousActions[7] + " " + continuousActions[8] + " " + continuousActions[9] + " " + continuousActions[10] + " " + continuousActions[11] + " " + continuousActions[12] + " " + continuousActions[13] + " " + continuousActions[14] + " " + continuousActions[15] + " " + continuousActions[16] + " " + continuousActions[17] + " " + continuousActions[18] + " " + continuousActions[19] + " " + continuousActions[20] + " " + continuousActions[21] + " " + continuousActions[22] + " " + continuousActions[23] + " " + continuousActions[24]);

            if (transform.Equals(UpperArm_L)) {
                
                bp.SetJointTargetRotation(continuousActions[0] * 1000, continuousActions[1] * 1000, continuousActions[2] * 1000);
                bp.SetJointStrength(continuousActions[3] * 1000);
            } else if (transform.Equals(LowerArm_L)) {
                
                bp.SetJointTargetRotation(continuousActions[4] * 1000, 0, 0);
                bp.SetJointStrength(continuousActions[5] * 1000);
            } else if (transform.Equals(UpperArm_R)) {
                
                bp.SetJointTargetRotation(continuousActions[6] * 1000, continuousActions[7] * 1000, continuousActions[8] * 1000);
                bp.SetJointStrength(continuousActions[9] * 1000);
            } else if (transform.Equals(LowerArm_R)) {
                // Debug.Log("continuous actions: " + continuousActions[10] + " " + continuousActions[11]);

                
                bp.SetJointTargetRotation(continuousActions[10] * 1000, 0, 0);
                bp.SetJointStrength(continuousActions[11] * 1000);
            } else if (transform.Equals(UpperLeg_L)) {
                // Debug.Log("upper leg left");
                // Debug.Log("continuous actions: " + continuousActions[12] + " " + continuousActions[13]);
                // Debug.Log("continuous actions: " + continuousActions[14]);
                bp.SetJointTargetRotation(continuousActions[12] * 1000, 0, continuousActions[13] * 1000);
                bp.SetJointStrength(continuousActions[14] * 1000);
            } else if (transform.Equals(LowerLeg_L)) {
                // Debug.Log("lower leg left");
                bp.SetJointTargetRotation(continuousActions[15] * 1000, 0, 0);
                bp.SetJointStrength(continuousActions[16] * 1000);
            } else if (transform.Equals(UpperLeg_R)) {
                // Debug.Log("upper leg right");
                bp.SetJointTargetRotation(continuousActions[17] * 1000, 0, continuousActions[18] * 1000);
                bp.SetJointStrength(continuousActions[19] * 1000);
            } else if (transform.Equals(LowerLeg_R)) {
                // 
                bp.SetJointTargetRotation(continuousActions[20] * 1000, 0, 0);
                bp.SetJointStrength(continuousActions[21] * 1000);
            } else if (transform.Equals(Head)) {
                // 
                bp.SetJointTargetRotation(continuousActions[22] * 1000, continuousActions[23] * 1000, 0);
                bp.SetJointStrength(continuousActions[24] * 1000);
            }
        }
        if (Opponent != null)
        {
            float currentDistanceToOpponent = Vector3.Distance(Body.transform.position, Opponent.position);
            if (currentDistanceToOpponent < previousDistanceToOpponent)
            {
                
                AddReward(10f); // Reward for getting closer
            }
            else if (currentDistanceToOpponent >= previousDistanceToOpponent)
            {   
                
                AddReward(-10f); // Negative reward for getting further
            }
            previousDistanceToOpponent = currentDistanceToOpponent;
        }
    }

    private void OnCollisionEnter(Collision collision) {
        if (collision.gameObject.name == "WallRight" || collision.gameObject.name == "WallLeft" || collision.gameObject.name == "WallWood" || collision.gameObject.name == "WallGrass") {
            AddReward(-2f); // Out of the arena
            EndEpisode();
        }
    }

    private void OnTriggerEnter(Collider other)
    {  
        if (other.gameObject.name == "WallRight" || other.gameObject.name == "WallLeft" || other.gameObject.name == "WallWood" || other.gameObject.name == "WallGrass") {
            AddReward(-2f); // Out of the arena
            EndEpisode();
        }
        if (other.TryGetComponent(out Wall wall)) {
            AddReward(-2f); // Out of the arena
            EndEpisode();
        }
    }

    public void OnBlockedHit() {
        AddReward(10f); // Hit the opposing shield or sword
    }

    public void OnSwordHitOpponent()
    {
        // Add a positive reward for hitting the opponent
        AddReward(1000f);
        EndEpisode();
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
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

