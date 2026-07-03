# Pre-Proposal

---

## 1. ชื่อโครงงาน

การถ่ายทอดทักษะการเดินข้ามสัณฐานวิทยาด้วย Latent Action World Models
*(Cross-Morphology Locomotion Transfer via Latent Action World Models)*

---

## 2. อาจารย์ที่ปรึกษา

— กรอกเอง

---

## 3. นักศึกษา

— กรอกเอง

---

## 4. ที่มาและความสำคัญ

ในปัจจุบัน การพัฒนาหุ่นยนต์ที่สามารถเดินได้ด้วย Reinforcement Learning (RL) ยังมีข้อจำกัดพื้นฐานที่สำคัญ นั่นคือ Policy ที่เทรนขึ้นมาจะผูกติดกับสัณฐานวิทยา (Morphology) ของหุ่นยนต์ตัวนั้นโดยตรง กล่าวคือ หาก Policy หนึ่งถูกเทรนให้หุ่นยนต์ขาสั้นเดินได้ Policy ดังกล่าวจะใช้ไม่ได้เลยกับหุ่นยนต์ขายาวหรือหุ่นยนต์ที่มีโครงสร้างร่างกายแตกต่างออกไป ต้องเริ่มเทรนใหม่ทั้งหมดทุกครั้ง ซึ่งเป็นอุปสรรคต่อการพัฒนาหุ่นยนต์หลายรูปแบบในเวลาและทรัพยากรที่จำกัด

งานวิจัยที่ผ่านมาพยายามแก้ปัญหานี้หลายแนวทาง เช่น DreamerV3 (Hafner et al., 2023) ที่ใช้ World Model ทำนายสถานะในอนาคตและเทรน Policy ผ่านการ Rollout จำลอง แต่โมเดลยังต้องการ Action Labels ที่ชัดเจนและยังผูกติดกับ Morphology เดิม และ LAC-WM (Latent Action Robot Foundation World Models) ที่เสนอการใช้ Latent Action Space รวมหุ่นยนต์หลายแบบในโดเมนการหยิบจับ (Manipulation) แต่ยังต้องพึ่ง Motion Labels ในขั้นตอน Pretraining ซึ่งเป็นไปไม่ได้ในโดเมน Locomotion ที่ไม่มีการบันทึก Ground-Truth Action โดยตรง

งานวิจัยนี้จึงเสนอแนวทางใหม่สำหรับโดเมน Locomotion โดยเฉพาะ โดยการเรียนรู้ **Latent Action Space ที่ไม่ขึ้นกับสัณฐานวิทยา (Morphology-Agnostic)** ผ่าน Inverse Dynamics Model (IDM) และ Forward Dynamics Model (FDM) ซึ่งเทรนได้จากวิดีโอ Simulation ล้วนๆ โดยไม่ต้องการ Action Labels ใดๆ ทั้งสิ้น แนวคิดหลักคือ IDM จะเรียนรู้ว่า "การเปลี่ยนแปลงใดเกิดขึ้นระหว่างเฟรม" และสกัดออกมาเป็น Latent Action $z_t$ ที่ควรจะสื่อถึงพฤติกรรมการเดิน (เช่น เดินตรง หรือ เลี้ยว) โดยไม่ขึ้นกับว่าหุ่นยนต์ตัวนั้นมีขาสั้นหรือยาว ซึ่งหาก $z_t$ มีคุณสมบัตินี้จริง ก็จะสามารถถ่ายทอดทักษะไปยังหุ่นยนต์ Morphology ใหม่ได้โดยใช้ข้อมูลน้อยกว่าการเทรนตั้งแต่ต้นอย่างมีนัยสำคัญ

---

## 5. วัตถุประสงค์

1. เพื่อออกแบบและเทรน Pipeline การสร้าง Latent Action Space แบบ Self-Supervised โดยใช้ IDM และ FDM บนข้อมูลวิดีโอจาก Simulation ของหุ่นยนต์แมลง (Stick Insect) 3 Morphology โดยไม่ต้องใช้ Action Labels ในขั้นตอน Pretraining
2. เพื่อพิสูจน์เชิงประจักษ์ว่า Latent Action $z_t$ ที่ได้มีคุณสมบัติ Morphology-Agnostic โดยใช้ Principal Component Analysis (PCA) วิเคราะห์ว่า $z_t$ จากหุ่นยนต์ต่าง Morphology ที่ทำพฤติกรรมเดียวกันเกาะกลุ่มร่วมกันหรือไม่
3. เพื่อทดสอบการถ่ายทอดความรู้ไปยัง Morphology ที่ไม่เคยเห็นในขั้นตอน Pretraining (Medium Leg) และวัดผลว่า World Model ที่ได้ช่วยลดปริมาณข้อมูลที่จำเป็นสำหรับ Morphology ใหม่ได้จริงหรือไม่

---

## 6. ขอบเขตการศึกษา

1. **ขอบเขตด้าน Platform:** ทำการทดลองใน Simulation เท่านั้น โดยสร้างหุ่นยนต์แมลง (Stick Insect) 3 แบบที่มีความยาวขาต่างกัน ได้แก่ ขาสั้น / ขากลาง / ขายาว ใน Environment จำลอง (เช่น MuJoCo หรือ IsaacGym) ยังไม่ครอบคลุมการทดลองกับหุ่นยนต์จริง
2. **ขอบเขตด้านเฟส:** ครอบคลุมเฉพาะ Phase 1 (Pretraining: IDM+FDM) และการตรวจสอบด้วย PCA เท่านั้น ยังไม่ครอบคลุม Phase 2 (Imagination RL หรือ Policy Training บน FDM) และ Phase 3 (Deployment)
3. **ขอบเขตด้านงาน (Task):** งานที่ศึกษาคือการเดินไปข้างหน้า (Forward Locomotion) เท่านั้น ไม่ครอบคลุม Manipulation, Climbing หรือ Terrain ที่ซับซ้อน
4. **ขอบเขตด้านข้อมูล:** ใช้วิดีโอจาก Simulation Camera (Third-Person View) ที่เห็นภาพรวมทั้ง Agent และ Environment ยังไม่ใช้วิดีโอสัตว์จริงจาก Dataset ภายนอก
5. **ขอบเขตการพิสูจน์:** การทดสอบ Morphology ใหม่ (Medium Leg) เป็นการพิสูจน์แบบ Interpolation (ค่าอยู่ในช่วงระหว่าง Short และ Long) ยังไม่ใช่ Extrapolation ไปยัง Morphology ที่อยู่นอกขอบเขต

---

## 7. ประโยชน์ที่คาดว่าจะได้รับ

1. ได้ Framework สำหรับการสร้าง Latent Action Space แบบ Self-Supervised สำหรับ Locomotion ที่ไม่ต้องการ Action Labels ซึ่งสามารถขยายไปใช้กับวิดีโอสัตว์จริง (เช่น Animal Kingdom Dataset) ในอนาคตได้โดยตรง
2. ได้แนวทางการถ่ายทอดทักษะการเดินไปยังหุ่นยนต์ที่มี Morphology ใหม่โดยลดปริมาณ Labeled Data ที่จำเป็นลงอย่างมีนัยสำคัญ เมื่อเปรียบเทียบกับการเทรน Policy ตั้งแต่ต้น
3. ได้วิธีการวัดและพิสูจน์คุณสมบัติ Morphology-Agnostic ของ Latent Space ด้วย PCA ซึ่งสามารถนำไปใช้เป็น Benchmark สำหรับงานวิจัยด้าน Cross-Morphology Transfer ในอนาคต

---

## 8. แผนการดำเนินงาน

| ลำดับ | แผนดำเนินงาน | ส.ค. | ก.ย. | ต.ค. | พ.ย. | ธ.ค. | ม.ค. |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | ออกแบบ System Overview และวาด Pipeline Diagram พร้อม Input/Output ชัดเจน | ✓ | | | | | |
| 2 | สร้าง Simulation Environment — Stick Insect 3 Morphology (MuJoCo/IsaacGym) | ✓ | | | | | |
| 3 | เก็บข้อมูล (Data Collection) จากหุ่นยนต์ขาสั้น + ขายาว (Train Set) | ✓ | ✓ | | | | |
| 4 | เทรน IDM + FDM (Phase 1 Pretraining) | | ✓ | | | | |
| 5 | PCA Validation — วิเคราะห์การจัดกลุ่มของ $z_t$ ตาม Behaviour vs Morphology | | ✓ | | | | |
| 6 | ทดสอบการ Transfer ไปยัง Medium Leg (Unseen Morphology) | | | ✓ | | | |
| 7 | วิเคราะห์ผลและเปรียบเทียบกับ Baseline | | | ✓ | ✓ | | |
| 8 | เขียนรายงานและจัดทำเอกสารฉบับสมบูรณ์ | | | | ✓ | ✓ | ✓ |
