# โน้ตอธิบายงานวิจัย — Cross-Morphology Locomotion via Latent Action World Models

---

## 1. ปัญหาคืออะไร

### ปัญหาพื้นฐาน

ในงานด้าน robot locomotion ปัจจุบัน เวลาเราจะสอน robot ให้เดินได้ เราต้องเทรน policy ซึ่งก็คือ neural network ที่รับ state ของ robot เป็น input แล้ว output ออกมาเป็น action เช่น จะหมุน joint ไหน เท่าไหร่ ปัญหาคือ policy นี้มัน specific กับ body ของ robot ตัวนั้นมากๆ เพราะมันเรียนรู้ว่า "ถ้าขา 4 ขาของฉันอยู่ในท่านี้ ให้ทำแบบนี้" ซึ่งถ้า body เปลี่ยนไป เช่น

- เปลี่ยนจาก 4 ขา เป็น 6 ขา
- เปลี่ยนสัดส่วนขาให้ยาวขึ้นหรือสั้นลง
- เปลี่ยน mass distribution

policy เดิมใช้ไม่ได้เลย ต้องเริ่มเทรนใหม่ตั้งแต่ต้น ซึ่งแต่ละครั้งใช้เวลาหลายชั่วโมงถึงหลายวัน และต้องใช้ computational resource จำนวนมาก

### ทำไมมันถึงเป็นปัญหาใหญ่

ในโลกจริง robot ไม่ได้มีแบบเดียว บริษัทต่างๆ ผลิต robot หลายรุ่น หลาย body configuration ถ้าทุก configuration ต้องเทรนใหม่ตั้งแต่ต้น มันไม่ scale ในทางปฏิบัติ นอกจากนี้ robot ที่ใช้งานจริงอาจเกิดความเสียหาย เช่น ขาหัก ซึ่งทำให้ morphology เปลี่ยนโดยกะทันหัน robot ต้องการ adapt ได้เร็ว ไม่ใช่รอ retrain ใหม่

### ชื่อเรียกของปัญหานี้ในวรรณกรรม

ปัญหานี้เรียกว่า **cross-morphology locomotion transfer** หรือบางครั้งเรียก **cross-embodiment generalization** คือการที่ความรู้เรื่องการเดินถ่ายทอดข้าม body ที่ต่างกันได้

---

## 2. Intuition — ทำไมถึงใช้ video สัตว์

### สัตว์แก้ปัญหานี้ได้แล้ว

ลองคิดดูว่าในธรรมชาติมีสัตว์ที่มี body แตกต่างกันอย่างสุดขั้ว

- แมลง มี 6 ขา ขาเล็กมาก น้ำหนักเบา
- สุนัข มี 4 ขา สัดส่วนกลางๆ
- ม้า มี 4 ขา ขายาว มวลมาก
- ตะขาบ มีขาหลายสิบขา

แต่ทุกตัวเดินได้ ทั้งหมดเรียนรู้หลักการเดียวกัน ได้แก่ การทรงตัวเมื่อยกขา การส่งน้ำหนักระหว่างขา การ recover เมื่อสะดุด และการ coordinate ขาหลายๆ ข้างพร้อมกัน ความรู้เหล่านี้ไม่ได้ขึ้นกับว่ามีขากี่ข้าง หรือขายาวแค่ไหน มันเป็น universal principle ของการเดินภายใต้ gravity และ physics จริงๆ

### ทำไม video ของสัตว์ถึงดีกว่า simulation

ปัจจุบัน robot learning ส่วนใหญ่ใช้ simulation เช่น MuJoCo หรือ Isaac Gym ในการเทรน แต่ simulation มีปัญหา เรียกว่า **sim-to-real gap** คือ physics ใน simulation มันเป็นแค่ approximation ของ physics จริง เช่น การสัมผัสระหว่างขากับพื้น การ deform ของวัสดุ และ friction จริงๆ ล้วน model ได้ไม่สมบูรณ์ใน simulation

ส่วน video ของสัตว์จากธรรมชาติ เช่น สารคดีธรรมชาติ หรือ YouTube เป็น physics จริงๆ 100% evolution optimize การเดินมาหลายร้อยล้านปีกับ physics จริง ไม่ใช่ simulation และ video เหล่านี้มีอยู่เยอะมากและฟรี

### Dataset ที่จะใช้

**Animal Kingdom** — พัฒนาโดย Singapore University of Technology and Design (SUTD) ปี 2022 เป็น dataset ที่รวบรวม video สัตว์กว่า 50 ชั่วโมง ครอบคลุม 850 species ใน 6 class หลัก ได้แก่ สัตว์เลี้ยงลูกด้วยนม นก สัตว์เลื้อยคลาน สัตว์สะเทินน้ำสะเทินบก ปลา และแมลง มี annotation สำหรับ action recognition และ pose estimation ซึ่งทำให้เป็น data source ที่เหมาะที่สุดสำหรับงานนี้

---

## 3. Stack ทางเทคนิค

### 3.1 World Model คืออะไร

**World Model** คือ neural network ที่เรียนรู้ว่า "ถ้า agent อยู่ใน state S แล้วทำ action A สภาพแวดล้อมจะเปลี่ยนไปเป็น S′ อย่างไร" พูดง่ายๆ คือมันเป็น internal model ของ physics ในหัว agent

ในแนวทางปกติที่ไม่มี world model agent ต้องลองทำจริงทุกครั้งเพื่อรู้ว่าเกิดอะไรขึ้น ซึ่งใช้ time และ interaction เยอะมาก แต่ถ้ามี world model agent สามารถ "จินตนาการ" ใน latent space ก่อนได้เลย คิดว่าถ้าทำแบบนี้จะเกิดอะไรขึ้น โดยไม่ต้องลองจริง ทำให้เรียนรู้ได้เร็วขึ้นมากและใช้ data น้อยลง

งานที่เป็นรากฐานสำคัญของ world model คือ **DreamerV3** พัฒนาโดย Danijar Hafner จาก Google DeepMind ปี 2023 ใช้ architecture ชื่อ **RSSM (Recurrent State Space Model)** ซึ่งเรียนรู้ latent space ของ environment แล้วให้ policy เรียนรู้จาก imagined rollouts ใน latent space นั้น DreamerV3 แสดงให้เห็นว่า world model เดียวสามารถ generalize ได้ใน 150+ tasks โดยไม่ต้องปรับ hyperparameter เลย รวมถึง locomotion tasks ใน MuJoCo ด้วย

### 3.2 ปัญหาเมื่อต้องการใช้ video สัตว์

World model แบบดั้งเดิมต้องการ action label คู่กับ observation เสมอ เช่น "ขณะนี้ขา joint 3 หมุน 15 องศาด้วย torque 20 Nm" แต่ใน video สัตว์ เราเห็นแค่ภาพที่สัตว์เคลื่อนที่ เราไม่รู้เลยว่า

- muscle ไหนออกแรงเท่าไหร่
- joint torque จริงๆ คืออะไร
- neural signal ที่ส่งไปยังกล้ามเนื้อมีค่าเท่าไหร่

action ซ่อนอยู่ใน movement ที่เห็น ทำให้ใช้ world model แบบ standard ไม่ได้

### 3.3 Latent Action Model คือทางออก

แนวคิดคือ แทนที่จะ require action label เราให้ model **อนุมาน** latent variable ขึ้นมาเองจาก observation สอง frame ติดกัน

มีสองส่วนหลัก

**IDM — Inverse Dynamics Model**
รับ observation สอง frame แล้ว encode ออกมาเป็น latent action z
```
IDM(pose_t, pose_t+1) → z_t
```
z_t คือ "สิ่งที่ทำให้เกิด transition นี้" โดยไม่ต้องรู้ว่ามันคือ torque หรือ muscle activation

**FDM — Forward Dynamics Model (World Model ตัวจริง)**
รับ state ปัจจุบันและ latent action แล้ว predict state ถัดไป
```
FDM(pose_t, z_t) → predicted pose_t+1
```

ทั้งสองส่วนเทรนพร้อมกันด้วย reconstruction loss คือ predicted pose_t+1 ต้องใกล้เคียงกับ pose_t+1 จริงๆ ให้มากที่สุด

วิธีนี้ทำงานได้เพราะ IDM ถูก "บังคับ" ให้ encode ข้อมูลที่จำเป็นและเพียงพอสำหรับ FDM ในการ predict state ถัดไปเท่านั้น ข้อมูลที่ไม่เกี่ยวกับ transition เช่น texture ของขน หรือสีพื้นหลัง จะไม่ถูก encode เข้าไปใน z เพราะมันไม่ช่วยให้ predict ได้ดีขึ้น

งานที่แสดงว่าแนวทางนี้ได้ผลในทางทฤษฎีคือ **"What Do Latent Action Models Actually Learn?"** โดย Zhang et al. จาก Microsoft Research ปี 2025 ตีพิมพ์ที่ NeurIPS 2025 งานนี้วิเคราะห์ด้วย linear model และแสดงให้เห็นว่า IDM objective บังคับให้ z encode เฉพาะ controllable changes เท่านั้น นอกจากนี้ยังแสดงว่า data augmentation และ data cleaning ช่วย enforce ให้ z capture การเปลี่ยนแปลงที่ควบคุมได้ ไม่ใช่ noise

### 3.4 งานที่ใช้ Latent Action Model ที่ผ่านมา

**Genie: Generative Interactive Environments**
พัฒนาโดย Google DeepMind ปี 2024 ตีพิมพ์ที่ ICML 2024
เป็นงานชิ้นแรกที่แสดงว่า latent action model สามารถเรียนรู้จาก Internet video ที่ไม่มี action label ได้ Genie train บน video เกม 2D platformer และเรียนรู้ discrete latent actions ที่ทำให้ user ควบคุม virtual environment ได้ โดยไม่เคยเห็น ground-truth action เลย ข้อจำกัดคือใช้ video เกม ซึ่ง clean และ controlled กว่า video สัตว์มากและ discrete action ไม่เหมาะกับ locomotion ที่ต้องการ continuous control

**LAPA: Latent Action Pretraining from Videos**
พัฒนาโดยทีมจาก University of Washington, Microsoft Research, และ NVIDIA ปี 2024 ตีพิมพ์ที่ ICLR 2025 ได้รับ Best Paper ที่ CoRL LangRob Workshop
LAPA ใช้ VQ-VAE เพื่อ discover discrete latent actions จาก Internet video แล้ว pretrain VLA (Vision-Language-Action) model โดยไม่ต้องมี robot action label เลย จากนั้น fine-tune บน robot data เพียงเล็กน้อย ผลลัพธ์คือ outperform state-of-the-art VLA ที่ train ด้วย labeled data ด้วยประสิทธิภาพ pretraining ที่ดีกว่า 30 เท่า ข้อจำกัดคือเน้น manipulation ไม่ใช่ locomotion

**CLAM: Continuous Latent Action Models**
พัฒนาโดยทีมจาก University of Southern California และ Google ปี 2025
ปรับปรุงจาก Genie และ LAPA โดยเปลี่ยนจาก discrete เป็น continuous latent actions เพราะ locomotion ต้องการ continuous joint control สอน latent IDM + latent FDM พร้อมกับ action decoder เพื่อให้ ground ได้ง่าย ข้อสำคัญคือ CLAM แสดงว่า continuous latent action ทำงานได้ดีกว่า discrete สำหรับ continuous control tasks ซึ่ง locomotion เป็น case ที่ตรงที่สุด

### 3.5 ทำไม z ถึง morphology-agnostic

นี่คือ core hypothesis ของงานเรา เมื่อ train บน biological video จาก 850+ species ที่มี body configuration แตกต่างกันอย่างสุดขั้ว model ถูกบังคับให้หา z ที่ explain transition ได้สำหรับทุก species พร้อมกัน

สิ่งที่ share ข้าม species ทั้งหมด ได้แก่ physics ของ locomotion เช่น การ shift center of mass ก่อนยกขา การ coordinate การก้าวขาเพื่อรักษา balance และการ recover เมื่อเสียหลัก สิ่งเหล่านี้จะถูก encode เข้าไปใน z เพราะมันอธิบาย transition ได้สำหรับทุก species

ส่วนสิ่งที่ body-specific เช่น ขนาดขา จำนวนข้อต่อ หรือ texture ของร่างกาย ไม่ได้ช่วยให้ predict transition ได้ดีขึ้นสำหรับ species อื่น z จึง pressure ให้ตัดสิ่งเหล่านี้ออก

**Embodiment Scaling Laws** — งานของ Ai et al. จาก National University of Singapore ปี 2025 ตีพิมพ์ที่ CoRL 2025 แสดงให้เห็นว่า diversity ของ morphology ใน training data มี scaling law คือยิ่งเทรนกับ morphology หลากหลาย ยิ่ง generalize ได้กว้างขึ้น และ embodiment diversity ให้ผลดีกว่าการเพิ่มปริมาณ data บน morphology เดิม งานนี้ทำบน simulation แต่ถ้า principle นี้ hold biological video ที่มี 850+ species น่าจะให้ diversity ที่ดีกว่า simulation dataset ใดๆ ที่มีอยู่

---

## 4. Pipeline ทั้งหมด

### Step 1 — เก็บ data

ใช้ Animal Kingdom dataset (850 species, 50 ชั่วโมง) เน้น clip ที่เห็นการเดินชัดเจน กรอง clip ที่มี occlusion หนักหรือ camera เคลื่อนไหวมากออก

### Step 2 — Pose Extraction

แปลง raw video เป็น keypoint sequence ด้วย **DeepLabCut** พัฒนาโดย Alexander Mathis จาก Harvard และ Tübingen University ปี 2018 ตีพิมพ์ใน Nature Neuroscience เป็น tool สำหรับ markerless pose estimation ของสัตว์ ใช้ transfer learning จาก ResNet ที่ pretrain บน ImageNet สามารถ estimate keypoints ได้แม่นยำระดับ human annotator ด้วย training data เพียง 200 frames

ผลที่ได้คือ sequence ของ pose แทนแต่ละ frame เช่น ตำแหน่ง (x, y) ของ joints แต่ละข้อในพิกัด body-centric

### Step 3 — Train Latent Action World Model

```
สำหรับแต่ละ consecutive frame pair (pose_t, pose_t+1):
   z_t = IDM(pose_t, pose_t+1)        ← อนุมาน latent action
   pose_pred = FDM(pose_t, z_t)       ← predict next state
   loss = ||pose_pred - pose_t+1||²   ← reconstruction loss
```

เทรนด้วย data จาก 850+ species พร้อมกัน model จะค่อยๆ เรียนรู้ latent space z ที่ capture locomotion structure ที่ share ข้าม species

### Step 4 — วิเคราะห์ Latent Space

ก่อน transfer ให้ visualize และวิเคราะห์ว่า z encode อะไรจริงๆ ด้วย t-SNE หรือ UMAP ดูว่า z organize ตาม gait type (walk, trot, gallop), morphology class (insect, quadruped), หรือ species หาก z cluster ตาม gait มากกว่า species แสดงว่า morphology-agnostic จริง

### Step 5 — Transfer ไปยัง Robot

เอา pretrained world model z ไป initialize policy learning ของ robot morphology ใหม่ใน simulation เช่น MuJoCo เปรียบเทียบ sample efficiency (ใช้ data น้อยกว่าเดิมแค่ไหน) และ asymptotic performance กับ baseline ที่ไม่ได้ pretrain บน biological video

---

## 5. งานที่มีอยู่และช่องว่าง

### กลุ่มที่ 1 — มี Latent Action WM แต่ใช้ Simulation

**LAC-WM (Latent Action Robot Foundation World Models)**
พัฒนาโดยทีมจาก Stanford University, Meta AI Research, และ FAIR ปี 2026 ตีพิมพ์ที่ ICLR 2026
ใช้ unified latent action space ข้าม robot embodiments หลายตัวใน simulation ไม่มี per-embodiment action label แต่ train บน simulation data ทั้งหมด แสดงว่า latent action space scale ได้กับจำนวน embodiment แต่ข้อจำกัดหลักคือ physics ยังมาจาก simulation อยู่

### กลุ่มที่ 2 — ใช้ Biological Video แต่ไม่มี Latent Action WM

**RLWAV (Reinforcement Learning from Wild Animal Videos)**
พัฒนาโดย Chane-Sane et al. จาก LAAS-CNRS, France ปี 2024
train video classifier บน animal video แล้วใช้ classification score เป็น reward signal ให้ robot เรียนรู้ใน simulation ไม่ต้องมี reference trajectory เลย แต่ classifier ยังต้องการ class label บน video และทดสอบแค่บน Solo quadruped ตัวเดียว ไม่ได้ cross morphology

**SLoMo (A General System for Legged Robot Motion Imitation from Casual Videos)**
พัฒนาโดยทีมจาก Carnegie Mellon University ปี 2023 ตีพิมพ์ใน IEEE Robotics and Automation Letters
แปลง casual video ของสัตว์และมนุษย์เป็น reference trajectory ผ่าน keypoint reconstruction และ trajectory optimization แล้ว track ด้วย MPC บน hardware ได้สำเร็จ แต่ต้องการ explicit trajectory ทุก video ไม่ได้เรียน shared representation

### ช่องว่างที่เห็น

| | Latent Action WM | Biological Video | Cross-Morphology |
|---|---|---|---|
| LAC-WM — Stanford, Meta, FAIR (ICLR 2026) | ✓ | ✗ simulation | ✓ |
| RLWAV — LAAS-CNRS France (2024) | ✗ classifier reward | ✓ | ✗ robot เดียว |
| SLoMo — CMU (RA-L 2023) | ✗ explicit trajectory | ✓ | ✗ robot เดียว |
| **งานที่เราจะทำ** | **✓** | **✓** | **✓** |

ยังไม่มีใครเอา latent action world model มา train บน biological video และ evaluate cross-morphology transfer โดยเฉพาะ นี่คือ design point ที่ว่างอยู่

---

## 6. ประโยคสรุปสำหรับบอกอาจารย์

> "เราจะ train latent action world model บน biological locomotion video จากสัตว์หลากหลาย species โดยไม่ต้องมี action label เลย โดย model จะอนุมาน latent variable z จาก observation สอง frame ติดกัน ซึ่ง z นี้จะ encode locomotion strategy ที่ share ข้าม morphology เช่น การทรงตัวและ gait rhythm โดยตัดสิ่งที่ body-specific ออกไป สมมติฐานคือ z นี้จะเป็น prior ที่ดีสำหรับ policy learning บน robot morphology ใหม่ ทำให้ transfer ข้าม body configuration ได้โดยไม่ต้อง retrain ใหม่ตั้งแต่ต้น"

---

## 7. คำถามที่อาจารย์น่าจะถาม

**"ทำไมการเดินของแมวถึงช่วย robot 6 ขาได้?"**

สิ่งที่ transfer ไม่ใช่ body plan แต่เป็น control strategy ว่าจะ shift weight ยังไงก่อนยกขา จะ sequence ขายังไงเพื่อรักษา balance และจะ recover จาก disturbance ยังไง สิ่งเหล่านี้เป็น physics-constrained ไม่ใช่ body-constrained เพราะ gravity ทำงานเหมือนกันกับทุก body plan หลักการ "ต้องมี center of mass อยู่เหนือ support polygon" ใช้ได้กับแมวและ robot 6 ขาเหมือนกัน

เรามีหลักฐาน empirical จาก SLoMo (CMU, 2023) และ RLWAV (LAAS-CNRS, 2024) ที่แสดงว่า biological locomotion signal ข้าม domain gap ไปถึง robot hardware ได้จริงแล้ว แม้จะยังไม่ใช่ latent action approach

**"Latent space จะไม่ learn visual artifact จาก video เหรอ?"**

นี่เป็น concern ที่ถูกต้อง และได้รับการตอบโดย Zhang et al. จาก Microsoft Research (NeurIPS 2025) ซึ่งวิเคราะห์ทางทฤษฎีว่า IDM objective บังคับให้ z encode เฉพาะ controllable changes เท่านั้น เพราะถ้า z encode สิ่งที่ไม่เกี่ยวกับ locomotion เช่น texture หรือ lighting ก็จะไม่ช่วยให้ FDM predict pose ถัดไปได้ดีขึ้น และ loss ก็จะไม่ลด นอกจากนี้ technique เช่น data augmentation และ data cleaning ช่วย enforce สิ่งนี้ได้ เราจะใช้ทั้งสองอย่างในงานนี้

นอกจากนี้ DiLA (2026) จาก Tsinghua University เสนอ content-structure disentanglement ที่แยก visual details ออกจาก locomotion structure อย่างชัดเจน ซึ่งเป็น architecture ที่น่าพิจารณาสำหรับงานเรา

**"Simulation ไม่พอเหรอ?"**

นั่นคือ null hypothesis ของงานเรา และเราไม่ได้อ้างว่า biological video ดีกว่าแน่นอน แต่มีเหตุผลที่ดีที่จะคิดว่ามันน่าจะให้ prior ที่ดีกว่าสำหรับ locomotion โดยเฉพาะ

Simulation ใช้ rigid-body physics ซึ่งเป็น approximation ในขณะที่สัตว์เดินบน real physics จริงๆ อีกทั้ง diversity ของ biological locomotion (850+ species) น่าจะให้ morphological diversity ที่กว้างกว่า simulation dataset ใดๆ ซึ่งสอดคล้องกับ embodiment scaling law ที่ Ai et al. จาก NUS พบว่า diversity matters มากกว่า quantity

คำตอบสุดท้ายจะมาจาก experiment ที่เปรียบเทียบ biological video pretraining กับ simulation pretraining บน benchmark เดียวกัน

**"Robot ที่ muscle-driven กับ joint-torque-driven มัน transfer กันได้จริงเหรอ?"**

นี่คือ counter-argument ที่แข็งที่สุด biological locomotion ใช้ muscle ที่ compliant และ elastic ในขณะที่ robot ส่วนใหญ่ใช้ rigid joint torque อาจเป็นไปได้ว่า z ที่ learn จาก biological video encode soft-body dynamics มากกว่า rigid-body strategy

อย่างไรก็ตาม RLWAV และ SLoMo แสดงให้เห็น empirically ว่า biological-to-robot transfer ไม่ได้ produce negative transfer ในทางปฏิบัติ เหตุผลที่น่าจะเป็นไปได้คือ high-level locomotion strategy เช่น gait sequence และ weight shift นั้น robust ต่อความแตกต่างของ actuator ส่วน low-level dynamics เช่น spring stiffness อาจ fine-tune ได้ในขั้นตอน adaptation บน robot morphology ใหม่

---

## 8. สิ่งที่ยังไม่รู้ (open questions สำหรับงานวิจัย)

1. Latent space z ที่ได้จาก biological video จะ organize ตาม gait type, species, หรือ morphology class อย่างไร
2. Multi-species biological diversity ให้ cross-morphology generalization ที่ดีกว่า simulation diversity จริงหรือไม่
3. Continuous latent action (CLAM-style) หรือ discrete (Genie-style) เหมาะกับ locomotion มากกว่ากัน
4. Species ไหนบ้างที่ให้ pose extraction ที่เชื่อถือได้ใน DeepLabCut และ species ไหนที่มี noise มากเกินไป
5. World model ที่ train บน biological video จะ generalize ไปยัง robot morphology ที่ไม่มีใน training data ได้ไกลแค่ไหน
