import pygame as pg
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

pg.init()
screen = pg.display.set_mode((1000, 560))
COLOR_INACTIVE = pg.Color('lightskyblue3')
COLOR_ACTIVE = pg.Color('dodgerblue2')
FONT = pg.font.Font(None, 32)
myfont = pg.font.SysFont('Comic Sans MS', 20)
white = (255, 255, 255)


class InputBox:

    def __init__(self, x, y, w, h, text=''):
        self.rect = pg.Rect(x, y, w, h)
        self.color = COLOR_INACTIVE
        self.text = text
        self.txt_surface = FONT.render(text, True, self.color)
        self.active = False

    def handle_event(self, event):
        if event.type == pg.MOUSEBUTTONDOWN:
            # If the user clicked on the input_box rect.
            if self.rect.collidepoint(event.pos):
                # Toggle the active variable.
                self.active = not self.active
            else:
                self.active = False
            # Change the current color of the input box.
            self.color = COLOR_ACTIVE if self.active else COLOR_INACTIVE
        if event.type == pg.KEYDOWN:
            if self.active:
                if event.key == pg.K_RETURN:
                    print(self.text)
                    self.text = ''
                elif event.key == pg.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                # Re-render the text.
                self.txt_surface = FONT.render(self.text, True, self.color)

    def update(self):
        # Resize the box if the text is too long.
        width = max(200, self.txt_surface.get_width()+10)
        self.rect.w = width

    def draw(self, screen):
        # Blit the text.
        screen.blit(self.txt_surface, (self.rect.x+5, self.rect.y+5))
        # Blit the rect.
        pg.draw.rect(screen, self.color, self.rect, 2)

def nametags():
    Gender = myfont.render('Gender', False, white)
    screen.blit(Gender, (10, 50))

    MP = myfont.render('MP', False, white)
    screen.blit(MP, (10, 100))

    DBMS = myfont.render('DBMS', False, white)
    screen.blit(DBMS, (10, 150))

    CN = myfont.render('CN', False, white)
    screen.blit(CN, (10, 200))

    TCS = myfont.render('TCS', False, white)
    screen.blit(TCS, (10, 250))

    OptionalName = myfont.render('Optional', False, white)
    screen.blit(OptionalName, (10, 300))

    OptionalMarks = myfont.render('Optional Marks', False, white)
    screen.blit(OptionalMarks, (10, 350))

    CGPAPrev = myfont.render('Prev CGPA', False, white)
    screen.blit(CGPAPrev, (10, 400))

def boxes():
    input_box_Gender = InputBox(170, 50, 140, 32)
    input_box_MP = InputBox(170, 100, 140, 32)
    input_box_DBMS = InputBox(170, 150, 140, 32)
    input_box_CN = InputBox(170, 200, 140, 32)
    input_box_TCS = InputBox(170, 250, 140, 32)
    input_box_OptionalName = InputBox(170, 300, 140, 32)
    input_box_OptionalMarks = InputBox(170, 350, 140, 32)
    input_box_CGPAPrev = InputBox(170, 400, 140, 32)
    input_boxes = [input_box_Gender, input_box_MP, input_box_DBMS, input_box_CN, input_box_TCS, input_box_OptionalMarks, input_box_OptionalName, input_box_CGPAPrev]
    return input_boxes


def trainModel(option, values=[1]):
    df = pd.read_csv("marks.csv")
    df = df[df['CGPA'] != 0]
    le = LabelEncoder()
    a = le.fit_transform(df.Gender)
    b = le.fit_transform(df['Optional Name'])
    df["Gender Encoded"] = a
    df["Elective encoded"] = b
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['Name', 'Gender', 'Optional Name', 'CGPA'], axis=1), df.CGPA, test_size=0.3, random_state=1)
    lr = LinearRegression()
    model = lr.fit(X_train, y_train)
    y_predict = lr.predict(X_test)
    mean_squared_error(y_test.values, y_predict) ** 0.5
    listOfFactors = ['MP', 'DBMS', 'CN', 'TCS', '(Optional Marks)', '(CGPA SEM 4)', 'Gender', 'Elective']
    expression = ""
    for i in range(8):
        if (lr.coef_[i] > 0 and i > 0):
            expression = expression + "+" + str(round(lr.coef_[i], 2)) + "*" + listOfFactors[i]
        else:
            expression = expression + str(round(lr.coef_[i], 2)) + "*" + listOfFactors[i]
    expression += '+' + str(round(lr.intercept_, 2))
    if option is 1:
        return [mean_squared_error(y_test.values, y_predict) ** 0.5, expression]
    else:
        gender, mp, dbms, cn, tcs, prevcgpa, optmarks,optional = values[0], int(values[1]), int(values[2]), int(values[3]), int(values[4]), float(values[7]), int(values[5]), values[6]
        # print(gender, mp, dbms, cn, tcs, prevcgpa)
        gender = 0 if gender is 'F' else 1
        optional = 0 if optional is 'AOS' else 1
        values = [mp, dbms, cn, tcs, optmarks,prevcgpa, gender, optional]
        values = np.asarray(values)
        print("Predicted", lr.predict([values]))
        return round((lr.predict([values])[0]), 2)


def load_screen(display="", duration = 2000):
    to_display = myfont.render(display, False, white)
    clock = pg.time.Clock()
    done = False
    load_bar = pg.Rect(350, 280, 320, 20)
    start_time = pg.time.get_ticks()
    while not done:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                done = True
        elapsed_time = pg.time.get_ticks() - start_time
        fill = (elapsed_time/duration) * 320
        progress = pg.Rect(350, 280, fill, 20)
        pg.draw.rect(screen, pg.Color('white'), progress)
        pg.draw.rect(screen, pg.Color('lightskyblue3'), load_bar, 2)
        screen.blit(to_display, (load_bar.x, load_bar.y - 40))
        pg.display.flip()
        clock.tick(30)
        if elapsed_time > duration:
            return


def main():
    clock = pg.time.Clock()
    input_boxes = boxes()
    done = False
    enter = pg.Rect(170, 450, 70, 32)
    predicted_score=""

    while not done:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                done = True
            for box in input_boxes:
                box.handle_event(event)
            if event.type == pg.MOUSEBUTTONDOWN:
                if enter.collidepoint(event.pos):
                    text_input = []
                    for box in input_boxes:
                        text_input.append(box.text)
                    print(text_input)
                    predicted_score = trainModel(2, text_input)

        for box in input_boxes:
            box.update()


        screen.fill((30, 30, 30))
        for box in input_boxes:
            box.draw(screen)

        txt_surface = myfont.render("Enter", True, pg.Color('lightskyblue3'))
        screen.blit(txt_surface, (enter.x + 5, enter.y + 5))
        pg.draw.rect(screen, pg.Color('lightskyblue3'), enter, 2)

        txt_surface = myfont.render("After Training, Error is: "+str(round(error, 2)), True, pg.Color('lightskyblue3'))
        screen.blit(txt_surface, (400, 40))
        txt_surface = myfont.render("Linear Regression equation is:", True, pg.Color('lightskyblue3'))
        screen.blit(txt_surface, (400, 80))
        txt_surface = myfont.render(expression[:55], True, pg.Color('lightskyblue3'))
        screen.blit(txt_surface, (400, 120))
        txt_surface = myfont.render(expression[56:], True, pg.Color('lightskyblue3'))
        screen.blit(txt_surface, (400, 160))
        txt_surface = myfont.render("The predicted score is: "+str(predicted_score), True, pg.Color('lightskyblue3'))
        screen.blit(txt_surface, (400, 200))

        nametags()
        pg.display.flip()
        clock.tick(30)


if __name__ == '__main__':
    error, expression= trainModel(1)
    print(error, expression)
    load_screen('Training model...', 5000)
    main()
    pg.quit()