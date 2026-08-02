จากการนำเสนอ **Proposal Presentation (หัวข้อ: Cross-Morphology Locomotion via Visual Action World Model)** ของไออุ่น ซึ่งมีอาจารย์และรุ่นพี่ร่วมให้ฟีดแบคและซักถามอย่างเข้มข้น สามารถสรุปคำแนะนำ ข้อสงสัย และประเด็นฟีดแบคทั้งหมดได้ดังนี้ครับ:

---

### **1. ฟีดแบคเรื่อง Impact และความคุ้มค่าของการใช้ Vision (อาจารย์แพรว / Prof. Preaw)**
* **จุดอ่อนของ Setup ปัจจุบัน (Over-simplified Setup):** อาจารย์แถวชี้ว่าการทดลองปัจจุบันตั้งอยู่บนพื้นเรียบ (Flat ground) และฟิกซ์มุมกล้องให้วิ่งตามหุ่น การใช้กล้องภายนอกมองแค่การขยับขาบนพื้นเรียบ ไม่ได้แสดงศักยภาพที่แท้จริงของ World Model เพราะมุมขาบนพื้นเรียบสามารถใช้เซ็นเซอร์ภายใน (Internal Sensors เช่น Joint Encoders) วัดได้ง่ายและตรงไปตรงมามากกว่าอยู่แล้ว.
* **ศักยภาพที่แท้จริงของ World Model (God-view):** ศักยภาพสูงสุดของการใช้กล้องภายนอก (External Vision) คือการมองเห็น **ข้อจำกัดจากสภาพแวดล้อม (Environment Constraints / Terrains)** เช่น ความขรุขระ ความสูงต่ำ (Elevation contrast), สโลป, หรือสิ่งกีดขวาง ร่วมกับท่าทางการวางตัวของหุ่น (Body posture/orientation).
* **คำถามสำคัญที่ต้องตอบให้ได้ (Key Justification):** ต้องพิสูจน์และตอบกรรมการให้ได้ว่า **"ทำไมการมองจากภายนอก (External Sensing) ถึงคุ้มค่าพอที่จะนำมาใช้ (Worth the setup complexity) เมื่อเทียบกับ Internal Sensors?"**.
* **ข้อเสนอแนะในการปรับปรุง:** ควรเลือก Task การทดสอบที่มีสภาพแวดล้อม/เทอร์เรนที่หลากหลายขึ้น (เช่น พื้นต่างระดับ หรือการใช้โค้ดสีที่พื้น) เพื่อดึงศักยภาพและโชว์ข้อดีของการใช้ Vision ให้เห็นเด่นชัด.

---

### **2. คำถามเรื่อง Generalization และการตั้งสมมติฐานระบบ (อาจารย์ตี๋ / Prof. Tee)**
* **ขอบเขตการ Generalize:** ปัจจุบันทดสอบบน Stick Insect ที่เปลี่ยนแค่ความยาวขา หากต้องการพิสูจน์ Cross-Morphology อย่างแท้จริง เสนอว่าควรขยายไปทดสอบกับหุ่นที่มีโครงสร้างต่างกันชัดเจน เช่น หุ่นหมา (Quadruped) หรือตุ๊กแก (Gecko) 0ะเป็นเซ็ทอัพที่เหมาะขึ้นหรือไม่
* **การกำหนดค่าพารามิเตอร์ (Pre-defined Inputs):** อาจารย์ตี๋ซักถามเรื่องมิติของ Action (เช่น $a \in \mathbb{R}^{18}$) ซึ่งยืนยันว่าเป็นค่าที่ Pre-defined ไว้ล่วงหน้า ไม่ได้ให้โมเดลเรียนรู้โครงสร้างบอร์ดี้ใหม่ทั้งหมดจากศูนย์.
* **Markov Assumption:** ปัจจุบันระบบมองข้อมูลแบบ Step-by-step (Markov state) คือดูแค่ State ถัดไป ($t+1$) ยังไม่ได้มองเป็น Sequential History ยาวๆ.
* **การทดสอบในขั้นตอน Test:** อาจารย์ตี๋ถามถึงกระบวนการทดสอบ ซึ่งไออุ่นอธิบายว่าจะนำ World Model ที่เทรนเสร็จแล้ว (ฟิกซ์ Checkpoint) ไปช่วยเทรน Policy ให้กับหุ่นตัวใหม่ที่ไม่เคยเห็นมาก่อน (ขากลาง - Medium leg) เพื่อดูว่า Loss curve ลู่ลงเร็วขึ้นหรือไม่.
* **ความยุ่งยากของมุมกล้องในโลกจริง:** การฟิกซ์กล้องให้เกาะติดหลังหุ่นยนต์ตลอดเวลาทำได้ยากในทางปฏิบัติ ต้องทบทวนการเซ็ตมุมกล้องใหม่ให้สมจริง.

---

### **3. ข้อเสนอแนะเรื่องการอธิบายประโยชน์ของ World Model (รุ่นพี่นาย & อาจารย์ตี๋)**
* **การชูจุดเด่นเรื่อง Sample Efficiency:** พี่นายแนะนำให้อธิบายให้ชัดเจนว่า World Model นำมาใช้ทำนายอนาคต (Imaginary Rollouts / Sequence Prediction) เพื่อนำมาประเมิน Reward และอัปเดต Policy ซึ่งช่วยประหยัดเวลาการลองผิดลองถูก (Trial-and-error) ใน Simulation และลด Sample Complexity.
* **ข้อควรระวังเรื่องคำว่า Prediction:** อาจารย์ตี๋ทักท้วงว่า เนื่องจากไปป์ไลน์มีการป้อนข้อมูล Future Frame เข้าไปใน Inverse Transition Model จึงต้องระวังและระบุให้ชัดเจนว่าส่วนไหนกันแน่ที่ทำหน้าที่ทำนายอนาคตโดยไม่เห็นข้อมูลล่วงหน้า.

---

### **4. คำถามเรื่องการ Fine-tune และการ Scaling ของระบบ (อาจารย์ตี๋ & ทีมงาน)**
* **การปรับแต่งโมเดลเมื่อเพิ่มหุ่นใหม่:** อาจารย์ถามว่าเมื่อนำ World Model ไปใช้กับหุ่นยนต์ร่างใหม่ จะต้อง Fine-tune ทั้งหมด หรือ Fine-tune แค่ส่วน Forward Transition.
* **ปัญหาการ Scaling:** อาจารย์ตี๋ตั้งข้อสังเกตว่าการ Fine-tune โมเดลขนาดใหญ่เมื่อเพิ่มข้อมูลใหม่เรื่อยๆ อาจทำได้ยากในทางปฏิบัติ (เปรียบเทียบกับปัญหาของ Autonomous Driving World Models เช่น Wayve ที่ไม่สามารถ Fine-tune ทั้งโมเดลได้ง่าย) จึงต้องมีแนวทางรับมือตรงนี้.

---

### **5. ไอเดียเสริมสำหรับอนาคต (Prof. Preaw)**
* **Internal Camera (On-body Camera):** อาจารย์แพรวเสนอไอเดียสนุกๆ สำหรับอนาคตว่า การติดกล้องไว้ที่ตัวหุ่น (First-person / Eye view) ภาพการสั่นสะเทือนและโมชันของกล้องขณะเคลื่อนที่ สามารถนำมาสกัดเป็น Latent Vector เพื่อประเมิน Body Attitude, Orientation, และความถี่การเคลื่อนที่ ซึ่งให้ข้อมูลมิติลึกกว่า IMU ปกติ.


P'nine
maybe clarifying whether it claims anything about learning time, or if the claim is the same as in the prior paper
might be good to explain how it speeds up model learning or improves performance, specifically
might be worth showing that the world model can explain the whole body and environment together, to predict good behavior overall.

P'Hap
Noted: the main point is about we used the bird eye views for the world model, while the tasks are too simplify.. i.e., walking on flat terrain.
If the terrain is constrained that mean the world model only predict the joint angle of the robots. Committees address that the world model with external sensing should have more potential then these simple tasks. Thinking about this krub