enum Gender {Male, Female}
public class Person {
    String name;
    int age;
    Gender gender;

    Person(){
        this.name = "홍길동";
        this.age = 20;
        this.gender = Gender.Male;
    }

    public Person(String name, int age, Gender gender) {
        this.name = name;
        this.age = age;
        this.gender = gender;
    }


    void showinfo(){
        System.out.println(String.format("이름:%s\n나이:%d\n성별:%s",this.name, this.age, this.gender));
    }

}

class Employee extends Person{
    int salary;

    Employee(){}
    Employee(String name, int age, Gender gender, int salary){
        super(name, age, gender);
        this.salary = salary;
    }

    @Override
    void showinfo() {
        System.out.printf("%s %d세 %s : %d\n", this.name, this.age, this.gender,this.salary);
    }
    void doWork(){
        System.out.println(this.name + "은/는 열심히 일합니다");
    }
}

class Prof extends Employee{
    Prof(String name, int age, Gender male, int salary) {
        super(name,age,male,salary);
    }

    void doWork(){
        System.out.println("학생을 가르친다");
    }
}
class Student extends Employee{
    Student(String name, int age, Gender gender, int salary) {
        super(name, age, gender, salary);
    }

    void doWork(){
        System.out.println("열심히 공부한다");
    }
}
class Manager extends Employee{
    Manager(String name, int age, Gender gender, int salary) {
        super(name, age, gender, salary);
    }
    void doWork(){
        System.out.println("학생과 교수를 관리한다");
    }
}

class Boss extends Person{
    Employee[] employees;
    int MAX_EMPLOY = 100;
    int n_curEmploy = 0;

    Boss(){
        this.name = "BOSS";
        this.age = 99;
        this.gender = Gender.Male;
        this.employees = new Employee[this.MAX_EMPLOY];
    }

    void hire(Employee e){
        if(this.n_curEmploy < this.MAX_EMPLOY){
            this.employees[n_curEmploy] = e;
            ++n_curEmploy;
        }
        else{
            System.out.println("더이상 고용 불가능");
        }
    }

    void makeEmployWork(){
        for(int i =0 ; i < this.n_curEmploy; ++i){
            employees[i].showinfo();
            employees[i].doWork();
        }
    }
}