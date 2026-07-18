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

ในปัจจุบัน การพัฒนาหุ่นยนต์ที่สามารถเดินได้ด้วย Reinforcement Learning (RL) ยังมีข้อจำกัดพื้นฐานที่สำคัญ นั่นคือ Policy ที่เทรนขึ้นมาจะผูกติดกับสัณฐานวิทยา (Morphology) ของหุ่นยนต์ตัวนั้นโดยตรง หาก Policy หนึ่งถูกเทรนให้หุ่นยนต์ขาสั้นเดินได้ Policy ดังกล่าวจะใช้ไม่ได้เลยกับหุ่นยนต์ขายาวหรือหุ่นยนต์ที่มีโครงสร้างร่างกายแตกต่างออกไป ต้องเริ่มเทรนใหม่ทั้งหมดทุกครั้ง หากสามารถเรียนรู้ "ภาษากลาง" ของการเคลื่อนที่ที่ใช้ร่วมกันข้ามสัณฐานวิทยาได้ ก็จะสามารถลดต้นทุนการพัฒนาหุ่นยนต์รูปแบบใหม่ได้อย่างมาก และนำทักษะที่สะสมไว้มาใช้ซ้ำข้ามรูปร่างหุ่นยนต์ที่แตกต่างกันได้โดยไม่ต้องเริ่มต้นใหม่

งานวิจัยที่ผ่านมาแก้ปัญหาได้บางส่วน แนวคิด World Model คือการสร้างโมเดลจำลองสภาพแวดล้อมภายในที่สามารถทำนายสถานะถัดไปจาก Action ที่กระทำ ซึ่งทำให้ Agent สามารถเรียนรู้ทักษะผ่านการ "จินตนาการ" หรือ Rollout ในโมเดลจำลองได้โดยไม่ต้องลองผิดลองถูกในสภาพแวดล้อมจริงทุกครั้ง DreamerV3 (Hafner et al., 2023) แสดงให้เห็นว่าแนวทางนี้ให้ผลดีกว่า 150 โดเมน แต่ยังผูกติดกับ Morphology เดิม ต้องเทรนใหม่ต่อหุ่นยนต์แต่ละแบบ ส่วน LAC-WM (Latent Action Robot Foundation World Models) ก้าวหน้าขึ้นอีกขั้นโดยเรียนรู้ Latent Action Space ที่แชร์ข้ามหุ่นยนต์หลายแบบในโดเมน Manipulation ได้ อย่างไรก็ตาม LAC-WM ยังต้องพึ่งวิดีโอที่มี Label กำกับในขั้นตอน Pretraining ซึ่งหมายความว่าไม่สามารถนำไปใช้กับวิดีโอสัตว์จริงที่ไม่มี Label ได้ และยังไม่เคยถูกนำมาประยุกต์กับโดเมน Locomotion เลย

งานวิจัยนี้จึงเสนอการนำแนวคิด Latent Action Space มาประยุกต์กับโดเมน Locomotion เป็นครั้งแรก โดยการเรียนรู้ **Latent Action Space ที่ไม่ขึ้นกับสัณฐานวิทยา (Morphology-Agnostic)** ผ่าน Inverse State-Transition Model (ITM) และ Forward State-Transition Model (FTM) Pipeline ใช้ข้อมูล Action ที่ Simulation บันทึกไว้โดยอัตโนมัติในการ Supervise z_t ผ่าน Motion Decoder (ตามแนวทางของ LAC-WM) เนื่องจากวิดีโอสัตว์จริงไม่มีข้อมูล Action กำกับ Simulation จึงเป็นแพลตฟอร์มที่เหมาะสมที่สุดในการพิสูจน์ว่า Latent Action Space ที่ได้มีคุณสมบัติ Morphology-Agnostic จริงหรือไม่ โดย ITM จะสกัด Latent Action $z_t$ ที่สื่อถึงพฤติกรรมการเดิน (เช่น เดินตรง หรือ เลี้ยว) โดยไม่ขึ้นกับรูปร่างของหุ่นยนต์ และการทดสอบด้วย PCA จะเป็นการยืนยันว่า z_t จากหุ่นยนต์ต่าง Morphology ที่ทำพฤติกรรมเดียวกันเกาะกลุ่มร่วมกันได้จริง

---

## 5. วัตถุประสงค์

1. เพื่อออกแบบและเทรน Pipeline การสร้าง Latent Action Space โดยใช้ Inverse State-Transition Model (ITM), Forward State-Transition Model (FTM) และ Motion Decoder บนข้อมูลวิดีโอและ Action Labels ที่ Simulation บันทึกไว้โดยอัตโนมัติ จากหุ่นยนต์แมลง (Stick Insect) 3 Morphology
2. เพื่อพิสูจน์เชิงประจักษ์ว่า Latent Action $z_t$ ที่ได้มีคุณสมบัติ Morphology-Agnostic โดยใช้ Principal Component Analysis (PCA) วิเคราะห์ว่า $z_t$ จากหุ่นยนต์ต่าง Morphology ที่ทำพฤติกรรมเดียวกันเกาะกลุ่มร่วมกันหรือไม่
3. เพื่อทดสอบการถ่ายทอดความรู้ไปยัง Morphology ที่ไม่เคยเห็นในขั้นตอน Pretraining (Medium Leg) และวัดผลว่า World Model ที่ได้ช่วยลดปริมาณข้อมูลที่จำเป็นสำหรับ Morphology ใหม่ได้จริงหรือไม่

---

## 6. ขอบเขตการศึกษา

1. **ขอบเขตด้าน Platform:** ทำการทดลองใน Simulation เท่านั้น โดยสร้างหุ่นยนต์แมลง (Stick Insect) 3 แบบที่มีความยาวขาต่างกัน ได้แก่ ขาสั้น / ขากลาง / ขายาว ใน Environment จำลอง (เช่น MuJoCo หรือ IsaacGym) ยังไม่ครอบคลุมการทดลองกับหุ่นยนต์จริง
2. **ขอบเขตด้านเฟส:** ครอบคลุมเฉพาะ Phase 1 (Pretraining: ITM + FTM) และการตรวจสอบด้วย PCA เท่านั้น ยังไม่ครอบคลุม Phase 2 (Imagination RL หรือ Policy Training บน FTM) และ Phase 3 (Deployment)
3. **ขอบเขตด้านงาน (Task):** งานที่ศึกษาคือการเดินไปข้างหน้า (Forward Locomotion) เท่านั้น ไม่ครอบคลุม Manipulation, Climbing หรือ Terrain ที่ซับซ้อน
4. **ขอบเขตด้านข้อมูล:** ใช้วิดีโอจาก Simulation Camera (Third-Person View) ที่เห็นภาพรวมทั้ง Agent และ Environment ยังไม่ใช้วิดีโอสัตว์จริงจาก Dataset ภายนอก
5. **ขอบเขตการพิสูจน์:** การทดสอบ Morphology ใหม่ (Medium Leg) เป็นการพิสูจน์แบบ Interpolation (ค่าอยู่ในช่วงระหว่าง Short และ Long) ยังไม่ใช่ Extrapolation ไปยัง Morphology ที่อยู่นอกขอบเขต

---

## 7. ประโยชน์ที่คาดว่าจะได้รับ

1. ได้ Framework สำหรับการสร้าง Latent Action Space สำหรับ Locomotion ที่ใช้ Action Labels จาก Simulation (ซึ่ง Auto-Log ได้โดยไม่ต้องมี Human Annotation) เป็น Supervision Signal ผ่าน Motion Decoder ซึ่งสามารถพิสูจน์คอนเซปต์และขยายทิศทางไปสู่การใช้วิดีโอสัตว์จริง (เช่น Animal Kingdom Dataset) ในอนาคตได้
2. ได้แนวทางการถ่ายทอดทักษะการเดินไปยังหุ่นยนต์ที่มี Morphology ใหม่โดยลดปริมาณ Labeled Data ที่จำเป็นลงอย่างมีนัยสำคัญ เมื่อเปรียบเทียบกับการเทรน Policy ตั้งแต่ต้น
3. ได้วิธีการวัดและพิสูจน์คุณสมบัติ Morphology-Agnostic ของ Latent Space ด้วย PCA ซึ่งสามารถนำไปใช้เป็น Benchmark สำหรับงานวิจัยด้าน Cross-Morphology Transfer ในอนาคต

---

## 8. แผนการดำเนินงาน

| ลำดับ | แผนดำเนินงาน | ส.ค. | ก.ย. | ต.ค. | พ.ย. | ธ.ค. | ม.ค. |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | ออกแบบ System Overview และวาด Pipeline Diagram พร้อม Input/Output ชัดเจน | ✓ | | | | | |
| 2 | สร้าง Simulation Environment — Stick Insect 3 Morphology (MuJoCo/IsaacGym) | ✓ | | | | | |
| 3 | เก็บข้อมูล (Data Collection) จากหุ่นยนต์ขาสั้น + ขายาว (Train Set) | ✓ | ✓ | | | | |
| 4 | เทรน ITM + FTM (Phase 1 Pretraining) | | ✓ | | | | |
| 5 | PCA Validation — วิเคราะห์การจัดกลุ่มของ $z_t$ ตาม Behaviour vs Morphology | | ✓ | | | | |
| 6 | ทดสอบการ Transfer ไปยัง Medium Leg (Unseen Morphology) | | | ✓ | | | |
| 7 | วิเคราะห์ผลและเปรียบเทียบกับ Baseline | | | ✓ | ✓ | | |
| 8 | เขียนรายงานและจัดทำเอกสารฉบับสมบูรณ์ | | | | ✓ | ✓ | ✓ |
