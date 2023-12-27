class Quizbrain:
    def __init__(self,q_list):
        self.question_no = 0
        self.question_list = q_list
        self.score = 0
        
    def still_has_question(self):
        return len(self.question_list) > self.question_no
         
        
        
    def next_question(self):
        question = self.question_list[self.question_no]
        self.question_no +=1
        
        user_ans = input(f"Q{self.question_no}."+ question.text +"? <True/False>")
        self.check_ans(user_ans,question.answer)
        
        
    def check_ans(self,user_ans,correct):
        if user_ans.lower() == correct.lower():
            print("Correct ans")
            self.score +=1
        else:
            print("Wrong ans")
            print(f"Your corrent score is {self.score}/{self.question_no}")
        print(f"Your corrent score is {self.score}/{self.question_no}")
        print("\n")
            
        
        
        
        
        