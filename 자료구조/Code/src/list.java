import java.util.ArrayList;
import java.util.LinkedList;

public class list {
    list(){};

    public void test(){
        Boss b = new Boss();
        Prof p1 = new Prof("교수1", 44, Gender.Male, 1000);
        Student s1 = new Student("학생1", 28, Gender.Female, 2000);
        Manager m1 = new Manager("매니저", 30, Gender.Male, 3000);

        ArrayList<Employee> al = new ArrayList<>();
        LinkedList<Employee> employees = new LinkedList<>();
        employees.add(p1);
        employees.add(s1);
        employees.add(m1);

        al.add(p1);
        al.add(s1);
        al.add(m1);

        for(Employee em : employees){
            em.doWork();
            em.doWork();
        }
    }


}
