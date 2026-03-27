# Presentation Script — Deep Compliant Control #3 (Midterm)
### 4 minutes total | ~30 seconds per slide

---

## Slide 1: Title (~10 sec)

"This is my midterm update on Deep Compliant Control for Quadruped Locomotion. The broader vision is making the case for quadrupeds as barrows — robots that can be physically guided by a human while carrying a payload over terrain where wheeled solutions fail."

---

## Slide 2: Problem Statement (~30 sec)

"Today's RL locomotion controllers are designed to be robust — they track a commanded velocity and treat any external force as a disturbance to reject. When a force persists, the robot fights it with high-frequency corrective torques that can exceed motor limits and damage the hardware.

The only way to interact with these robots is through a joystick. There's no mechanism for a human to physically guide the robot.

For the barrow use case this gets worse: the robot must simultaneously handle human guidance forces, payload dynamics shifting its centre of mass, and terrain disturbances — and current controllers don't distinguish between any of these. Robust is not enough."

---

## Slide 3: Project Goal (~30 sec)

"The thesis addresses four progressive challenges. First, compliant control through proprioceptive force estimation — no force-torque sensor, the robot estimates forces from joint history alone and adjusts its velocity accordingly. Second, extending this to uneven terrain and inclines, which is where I'm currently studying the influence of gravity on the force estimates. Third, adding a payload and accounting for its effects on the estimation. And fourth, demonstrating the terrain advantage — scenarios where a compliant payload-carrying quadruped offers more versatility than a wheeled barrow.

The compliance level is adjustable at deployment through two parameters, alpha and beta, without any retraining."

---

## Slide 4: Literature & Positioning (~35 sec)

"Existing work falls into two camps. Model-based methods like Kang et al. are limited to small perturbations under 50 Newtons and predefined gaits. On the learning side, Hartmann et al. handle transient pushes but fail under persistent force. Li et al. achieve compliance through direct torque output, but it's fixed — you can't track velocity commands under force.

The two works I build on most directly are HAC-LOCO, which introduces the hierarchical architecture with a frozen low-level policy and a learned force estimator, and SAC-Loco, which adds adjustable compliance levels. For the payload extension, Beyond Robustness by Chang et al. provides the path — they estimate load characteristics from proprioception using an architecture compatible with mine.

A concurrent project by Cao et al. at my lab is working on a related problem but from the opposite direction — their focus is payload transport with a leader robot, where compliance is secondary. My focus is compliance as the primary goal, with payload as the extension."

---

## Slide 5: Methodology — Architecture (~40 sec)

"The framework has two training stages, both now complete. In Stage 1, I train a robust locomotion policy alongside a force estimation autoencoder. The policy takes 60 dimensions of proprioceptive input and outputs 12 joint targets through a 512-256-128 MLP. The estimator takes a 20-step history, encodes it to a 64-dimensional latent, branches into a force head predicting 3D force, and a decoder that reconstructs the next observation for regularisation. The critic sees privileged ground-truth forces during training that the policy never accesses.

In Stage 2, I freeze the low-level policy and train a compliance mapping on top. It follows a three-phase curriculum: first pure locomotion, then forces activate and the estimator co-trains, then the compliance mapping turns on. The rule is simple: modified velocity equals the commanded velocity plus k times the estimated force, where k depends on the threshold alpha and impedance beta — both tunable at deployment.

The next step architecturally is the payload extension: adding payload randomisation in simulation and investigating whether the existing autoencoder can accommodate a load estimation head to decompose forces."

---

## Slide 6: Results So Far (~30 sec)

"Both stages are complete and deployed on the real Go2. The force estimator achieves about 6 to 7 degrees median angular error and 2.7 Newtons mean absolute error. The compliant policy is working on the real robot — I've tested it on grass, small gravel, and slight inclines without issues. The robot resists small disturbances and yields to sustained human pulls with a tunable compliance coefficient.

The current challenge is gravity and payloads. On inclines, gravity projects onto the force estimate, so the estimator confuses the slope with an applied force. Adding a payload makes this worse by shifting the centre of mass. The open question now is how to decompose the estimated force into human-applied versus gravity versus payload-induced components."

---

## Slide 7: Timeline & Next Steps (~25 sec)

"I'm in week 6 of a 13-week timeline. Setup, estimator, and compliance are all done — ahead of the original schedule. I'm now entering the gravity and payload phase, which runs through week 9.

The immediate next steps are: characterising the gravity influence on force estimation across different incline angles, adding payload randomisation in Isaac Sim with varying mass and centre-of-mass offsets, and investigating force decomposition strategies — whether the existing autoencoder can learn to separate these components or whether a dedicated load head is needed.

Weeks 10 to 13 are for quantitative evaluation, the terrain comparison experiments, and thesis writing."

---

## Slide 8: Summary (~10 sec)

"To summarise: RL controllers that reject all forces are insufficient for human-robot interaction and payload transport. I've built a hierarchical compliance framework that's working on the real Go2 across grass, gravel, and inclines. The current frontier is handling payloads and gravity effects — decomposing what the estimator sees into human-applied versus environment-induced forces. The end goal is making the case for quadrupeds as barrows."
