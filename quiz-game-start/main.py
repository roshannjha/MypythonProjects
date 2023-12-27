from question_model import Question
from data import question_data
from quiz_brain import Quizbrain

question_bank = []

for q in question_data:
    text = q['text']
    ans = q['answer']
    qn = Question(text,ans)
    question_bank.append(qn)
    
#print(question_bank[0].text)
qz_b = Quizbrain(question_bank)

while qz_b.still_has_question():
    qz_b.next_question()
    
print("You have completed the quiz")
print(f"Your final score was:  {qz_b.score}/{qz_b.question_no}")
            
            
        