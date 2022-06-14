import java.util.Scanner;

class TV{
    protected String manufacture;
    protected int makeYear;
    protected int inch;
    public TV(){
        this("LG", 2022, 28);
    }
    public TV(String _manufacture){
        this(_manufacture, 2022, 28);
    }
    public TV(String _manufacture, int _makeYear){
        this(_manufacture, _makeYear, 28);

    }
    public TV(String _manufacture, int _makeYear, int _inch){
        this.manufacture = _manufacture;
        this.makeYear =_makeYear;
        this.inch = _inch;
    }
    public void show(){
        System.out.println(this.manufacture + "에서 만든 " + this.makeYear + "년형 " + this.inch + "인치 TV");
    }
}
class Grade{
    private int mat;
    private int sci;
    private int eng;

    public Grade(int _mat, int _sci, int _eng) {
        this.mat = _mat;
        this.sci = _sci;
        this.eng = _eng;
    }
    public int average(){
        return Math.round((mat + sci + eng) / 3.0f);
    }
}
public class EXAM {
    public static void e1(){
        TV myTV = new TV("LG", 2017, 32);
        myTV.show();
    }
    public static void e2(){
        Scanner sc = new Scanner(System.in);
        Grade me = new Grade(sc.nextInt(), sc.nextInt(), sc.nextInt());
        System.out.println("평균은 " + me.average());
    }
    public static void main(String[] args){
        e2();
    }
}
