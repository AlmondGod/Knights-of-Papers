using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Unity.VisualScripting;

//this is a knight that has a sword and shield. Using its vision, its goal is to hit the opposing knight with the sword. However, if it hits the opposing knights shield or sword, this doesnt count as a hit. The knight can also block the opposing knights sword with its shied or sword. The knight can also move around the arena, as can the opposing knight. This knight starts across from but far from
//the opposing knight. The sword and shielf are attached to this knights lower right and lower left arms respectively. im using a camera for vision, so the knight can see the opposing knight we can find the opposing knights sword shield and body objects since this knight the other knigth and the floor are in a prefab
public class KnightofWood : Agent {
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
        m_JdController.maxJointForceLimit = 100000f;
        m_JdController.maxJointSpring = 100000f;
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
        transform.position = new Vector3(0, 2.38f, -4);  // Adjust the y-value as needed for correct height above the floor
        // transform.rotation = Quaternion.Euler(0, 0, 0);  // Adjust the Euler angles as needed to face the opponent
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
        if(Opponent == null) {
            Debug.Log("Opponent is null");
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
            AddReward(0.01f);
        }
    }

    //the knight can both move and rotate its upper and lower right and left arms
    //as well as its right and left foot and knee
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
         var bpDict = m_JdController.bodyPartsDict;
        var continuousActions = actionBuffers.ContinuousActions;
        Debug.Log("continuous actions length: " + continuousActions.Length);

        foreach (var transform in bpDict.Keys)
        {
            //if bp is head dont rotate on the z axis
            //if bp is lower arm or lower leg dont rotate on z or y axis
            //if bp is upper leg dont rotate on y axis
            var bp = bpDict[transform];

             if (transform.Equals(UpperArm_L)) {
                Debug.Log("UpperArm_L");
                bp.SetJointTargetRotation(continuousActions[0], continuousActions[1], continuousActions[2]);
                bp.SetJointStrength(continuousActions[3]);
            } else if (transform.Equals(LowerArm_L)) {
                Debug.Log("LowerArm_L");
                bp.SetJointTargetRotation(continuousActions[4], 0, 0);
                bp.SetJointStrength(continuousActions[5]);
            } else if (transform.Equals(UpperArm_R)) {
                Debug.Log("UpperArm_R");
                bp.SetJointTargetRotation(continuousActions[6], continuousActions[7], continuousActions[8]);
                bp.SetJointStrength(continuousActions[9]);
            } else if (transform.Equals(LowerArm_R)) {
                Debug.Log("LowerArm_R");
                bp.SetJointTargetRotation(continuousActions[10], 0, 0);
                bp.SetJointStrength(continuousActions[11]);
            } else if (transform.Equals(UpperLeg_L)) {
                Debug.Log("UpperLeg_L");
                bp.SetJointTargetRotation(continuousActions[12], 0, continuousActions[13]);
                bp.SetJointStrength(continuousActions[14]);
            } else if (transform.Equals(LowerLeg_L)) {
                Debug.Log("LowerLeg_L");
                bp.SetJointTargetRotation(continuousActions[15], 0, 0);
                bp.SetJointStrength(continuousActions[16]);
            } else if (transform.Equals(UpperLeg_R)) {
                Debug.Log("UpperLeg_R");
                bp.SetJointTargetRotation(continuousActions[17], 0, continuousActions[18]);
                bp.SetJointStrength(continuousActions[19]);
            } else if (transform.Equals(LowerLeg_R)) {
                Debug.Log("LowerLeg_R");
                bp.SetJointTargetRotation(continuousActions[20], 0, 0);
                bp.SetJointStrength(continuousActions[21]);
            } else if (transform.Equals(Head)) {
                Debug.Log("Head");
                bp.SetJointTargetRotation(continuousActions[22], continuousActions[23], 0);
                bp.SetJointStrength(continuousActions[24]);
            }
        }
        if (Opponent != null)
        {
            float currentDistanceToOpponent = Vector3.Distance(Body.transform.position, Opponent.position);
            if (currentDistanceToOpponent < previousDistanceToOpponent)
            {
                Debug.Log("positive closer reward");
                AddReward(0.01f); // Reward for getting closer
            }
            else if (currentDistanceToOpponent >= previousDistanceToOpponent)
            {   
                Debug.Log("further reward");
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
        AddReward(10f); // Hit the opposing shield or sword
    }

    public void OnSwordHitOpponent()
    {
        // Add a positive reward for hitting the opponent
        AddReward(100f);
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
        //in total there should be 25
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

        //strength
        continuousActions[16] = Input.GetKey(KeyCode.J) ? 1000f : 0.0f; 
        continuousActions[17] = Input.GetKey(KeyCode.K) ? 1000f : 0.0f; 
        continuousActions[18] = Input.GetKey(KeyCode.L) ? 1000f : 0.0f; 
        continuousActions[19] = Input.GetKey(KeyCode.M) ? 1000f : 0.0f;
        continuousActions[20] = Input.GetKey(KeyCode.N) ? 1000f : 0.0f;
        continuousActions[21] = Input.GetKey(KeyCode.O) ? 1000f : 0.0f;
        continuousActions[22] = Input.GetKey(KeyCode.P) ? 1000f : 0.0f;
        continuousActions[23] = Input.GetKey(KeyCode.Q) ? 1000f : 0.0f;
        continuousActions[24] = Input.GetKey(KeyCode.R) ? 1000f : 0.0f;
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

public class BodyPart 
{
        [Header("Body Part Info")][Space(10)] public ConfigurableJoint joint;
        public Rigidbody rb;
        [HideInInspector] public Vector3 startingPos;
        [HideInInspector] public Quaternion startingRot;

        [Header("Ground & Target Contact")]
        [Space(10)]
        public GroundContact groundContact;

        [HideInInspector] public JointDriveController thisJdController;

        [Header("Current Joint Settings")]
        [Space(10)]
        public Vector3 currentEularJointRotation;

        [HideInInspector] public float currentStrength;
        public float currentXNormalizedRot;
        public float currentYNormalizedRot;
        public float currentZNormalizedRot;

        [Header("Other Debug Info")]
        [Space(10)]
        public Vector3 currentJointForce;

        public float currentJointForceSqrMag;
        public Vector3 currentJointTorque;
        public float currentJointTorqueSqrMag;
        public AnimationCurve jointForceCurve = new AnimationCurve();
        public AnimationCurve jointTorqueCurve = new AnimationCurve();

        /// <summary>
        /// Reset body part to initial configuration.
        /// </summary>
        public void Reset(BodyPart bp)
        {
            bp.rb.transform.position = bp.startingPos;
            bp.rb.transform.rotation = bp.startingRot;
            bp.rb.velocity = Vector3.zero;
            bp.rb.angularVelocity = Vector3.zero;
        }

        /// <summary>
        /// Apply torque according to defined goal `x, y, z` angle and force `strength`.
        /// </summary>
        public void SetJointTargetRotation(float x, float y, float z)
        {
            x = (x + 1f) * 0.5f;
            y = (y + 1f) * 0.5f;
            z = (z + 1f) * 0.5f;

            var xRot = Mathf.Lerp(joint.lowAngularXLimit.limit, joint.highAngularXLimit.limit, x);
            var yRot = Mathf.Lerp(-joint.angularYLimit.limit, joint.angularYLimit.limit, y);
            var zRot = Mathf.Lerp(-joint.angularZLimit.limit, joint.angularZLimit.limit, z);

            currentXNormalizedRot =
                Mathf.InverseLerp(joint.lowAngularXLimit.limit, joint.highAngularXLimit.limit, xRot);
            currentYNormalizedRot = Mathf.InverseLerp(-joint.angularYLimit.limit, joint.angularYLimit.limit, yRot);
            currentZNormalizedRot = Mathf.InverseLerp(-joint.angularZLimit.limit, joint.angularZLimit.limit, zRot);

            joint.targetRotation = Quaternion.Euler(xRot, yRot, zRot);
            currentEularJointRotation = new Vector3(xRot, yRot, zRot);
        }

        public void SetJointStrength(float strength)
        {
            var rawVal = (strength + 1f) * 0.5f * thisJdController.maxJointForceLimit;
            var jd = new JointDrive
            {
                positionSpring = thisJdController.maxJointSpring,
                positionDamper = thisJdController.jointDampen,
                maximumForce = rawVal
            };
            joint.slerpDrive = jd;
            currentStrength = jd.maximumForce;
        }
}

    public class JointDriveController : MonoBehaviour
    {
        [Header("Joint Drive Settings")]
        [Space(10)]
        public float maxJointSpring;

        public float jointDampen;
        public float maxJointForceLimit;

        [HideInInspector] public Dictionary<Transform, BodyPart> bodyPartsDict = new Dictionary<Transform, BodyPart>();

        [HideInInspector] public List<BodyPart> bodyPartsList = new List<BodyPart>();
        const float k_MaxAngularVelocity = 50.0f;

        /// <summary>
        /// Create BodyPart object and add it to dictionary.
        /// </summary>
        public void SetupBodyPart(Transform t)
        {
            var bp = new BodyPart
            {
                rb = t.GetComponent<Rigidbody>(),
                joint = t.GetComponent<ConfigurableJoint>(),
                startingPos = t.position,
                startingRot = t.rotation
            };
            bp.rb.maxAngularVelocity = k_MaxAngularVelocity;

            if(bp.joint) {
                bp.joint.xMotion = ConfigurableJointMotion.Locked;
                bp.joint.yMotion = ConfigurableJointMotion.Locked;
                bp.joint.zMotion = ConfigurableJointMotion.Locked;
            } 
            
            // Add & setup the ground contact script
            bp.groundContact = t.GetComponent<GroundContact>();
            if (!bp.groundContact)
            {
                bp.groundContact = t.gameObject.AddComponent<GroundContact>();
                bp.groundContact.agent = gameObject.GetComponent<Agent>();
            }
            else
            {
                bp.groundContact.agent = gameObject.GetComponent<Agent>();
            }

            if (bp.joint)
            {
                var jd = new JointDrive
                {
                    positionSpring = maxJointSpring,
                    positionDamper = jointDampen,
                    maximumForce = maxJointForceLimit
                };
                bp.joint.slerpDrive = jd;
            }

            bp.thisJdController = this;
            bodyPartsDict.Add(t, bp);
            bodyPartsList.Add(bp);
        }

        public void GetCurrentJointForces()
        {
            foreach (var bodyPart in bodyPartsDict.Values)
            {
                if (bodyPart.joint)
                {
                    bodyPart.currentJointForce = bodyPart.joint.currentForce;
                    bodyPart.currentJointForceSqrMag = bodyPart.joint.currentForce.magnitude;
                    bodyPart.currentJointTorque = bodyPart.joint.currentTorque;
                    bodyPart.currentJointTorqueSqrMag = bodyPart.joint.currentTorque.magnitude;
                    if (Application.isEditor)
                    {
                        if (bodyPart.jointForceCurve.length > 1000)
                        {
                            bodyPart.jointForceCurve = new AnimationCurve();
                        }

                        if (bodyPart.jointTorqueCurve.length > 1000)
                        {
                            bodyPart.jointTorqueCurve = new AnimationCurve();
                        }

                        bodyPart.jointForceCurve.AddKey(Time.time, bodyPart.currentJointForceSqrMag);
                        bodyPart.jointTorqueCurve.AddKey(Time.time, bodyPart.currentJointTorqueSqrMag);
                    }
                }
            }
        }

        
    }
    
    public class GroundContact : MonoBehaviour
    {
        [HideInInspector] public Agent agent;

        [Header("Ground Check")] public bool agentDoneOnGroundContact; // Whether to reset agent on ground contact.
        public bool penalizeGroundContact; // Whether to penalize on contact.
        public float groundContactPenalty; // Penalty amount (ex: -1).
        public bool touchingGround;
        const string k_Ground = "ground"; // Tag of ground object.

        /// <summary>
        /// Check for collision with ground, and optionally penalize agent.
        /// </summary>
        void OnCollisionEnter(Collision col)
        {
            if (col.transform.CompareTag(k_Ground))
            {
                touchingGround = true;
                if (penalizeGroundContact)
                {
                    Debug.Log("called penalize ground contact");
                    agent.SetReward(groundContactPenalty);
                }

                if (agentDoneOnGroundContact)
                {
                    agent.EndEpisode();
                }
            }
        }

        /// <summary>
        /// Check for end of ground collision and reset flag appropriately.
        /// </summary>
        void OnCollisionExit(Collision other)
        {
            if (other.transform.CompareTag(k_Ground))
            {
                touchingGround = false;
            }
        }
    }

    public class OrientationCubeController : MonoBehaviour
    {
        //Update position and Rotation
        public void UpdateOrientation(Transform rootBP, Transform target)
        {
            var dirVector = target.position - transform.position;
            dirVector.y = 0; //flatten dir on the y. this will only work on level, uneven surfaces
            var lookRot =
                dirVector == Vector3.zero
                    ? Quaternion.identity
                    : Quaternion.LookRotation(dirVector); //get our look rot to the target

            //UPDATE ORIENTATION CUBE POS & ROT
            transform.SetPositionAndRotation(rootBP.position, lookRot);
        }
    }
