import java.util.InputMismatchException;
import java.util.Scanner;

class TV{
    protected String manufacture;
    protected int makeYear;
    protected int inch;
    private int size;

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
class ColorTv extends TV{
    private int colors;
    private int size;
    public ColorTv(int _color, int _size){
        super();
        this.colors = _color;
        this.size = _size;
    }
}
class A{
    private int a;
    protected A(int i){this.a=i;}
    public A(){a = 10;}
}
class B extends A{
    private int b;
    public B(){
        super();
        b = 0;
    }
}
class Grade{
    private int mat = 0;
    private int sci = 0;
    private int eng = 0;

    public Grade(int _mat, int _sci, int _eng) {
        this.mat = _mat;
        this.sci = _sci;
        this.eng = _eng;
    }
    public int average(){
        return Math.round((mat + sci + eng) / 3.0f);
    }
}
class Song{
    private final String title;
    private final String artist;
    private final String country;
    private final int year;
    public Song(){
        this("", "", "", 0);
    }
    public Song(String _title, String _artist, String _country, int _year){
        this.title = _title;
        this.artist = _artist;
        this.country = _country;
        this.year = _year;
    }
    public void show(){
        String sb = this.year + "년 " +
                this.country + "의 " +
                this.artist + "가 부른 " +
                this.title;
        System.out.println(sb);
    }
}
class MonthSchedule{
    class Day{
        private String todo;
        public Day(){}
        public void set(String _todo){
            this.todo = _todo;
        }
        public void show(){
            if(todo == null) System.out.println("일정이 없습니다.");
            else System.out.println(this.todo);
        }
        public void delDay(){
            this.todo = null;
        }
    }
    public Day[] schedule;
    public MonthSchedule(){
        schedule = new Day[30];
        for(int i = 0; i < 30; i++){
            schedule[i] = new Day();
        }
    }
    public void run(){
        while (true) {
            System.out.print("할일(입력:1, 보기:2, 끝내기:3) >>");
            Scanner sc = new Scanner(System.in);
            int input = sc.nextInt();
            if(input == 1){
                this.input();
            }
            else if(input == 2){
                this.view();
            }
            else if(input == 3){
                this.finish();
                break;
            }
        }
    }
    private void input(){
        Scanner sc = new Scanner(System.in);
        while (true) {
            System.out.print("날짜(1~30) >>");
            int index = sc.nextInt();
            if(index >= 1 && index <= 30){
                System.out.print("할일?");
                sc = new Scanner(System.in);
                String contents = sc.nextLine();
                this.schedule[index-1].set(contents);
                break;
            }
        }
    }
    private void view(){
        Scanner sc = new Scanner(System.in);
        while(true){
            System.out.print("날짜(1~30) >>");
            int index = sc.nextInt();
            if(index >= 1 && index <= 30){
                this.schedule[index-1].show();
                break;
            }
        }
    }
    private void finish(){
        for(int i = 0; i < 30; i++){
            this.schedule[i].delDay();
            this.schedule[i] = null;
        }
        this.schedule = null;
        System.out.println("프로그램을 종료합니다.");
    }
}
class Pen{
    private int amount;
    public int getAmount(){return this.amount;}
    public void setAmount(int _amount){this.amount = _amount;}
}
class SharpPencil extends Pen{
    private int width;
    public int getWidth(){return this.width;}
    public void setWidth(int _width){this.width = _width;}
}
class ColorPen extends Pen{
    private String color;
    public String getColor(){return this.color;}
    public void setColor(String _color){this.color = _color;}
}
class BallPen extends ColorPen{}
class FountainPen extends ColorPen{
    public void refill(int n){setAmount(n);}
}
abstract class OddDetector{
    protected int n;
    public OddDetector(int _n){
        this.n = _n;
    }
    public abstract boolean isOdd();
}
public class C extends OddDetector{
    public C(int _n){
        super(_n);
    }
    public boolean isOdd(){
        if(this.n % 2 == 0)
            return false;
        else return true;
    }
}

public class EXAM {
    public void silsoup1(){
        TV myTV = new TV("LG", 2017, 32);
        myTV.show();
    }
    public void silsoup2(){
        Scanner sc = new Scanner(System.in);
        Grade me = new Grade(sc.nextInt(), sc.nextInt(), sc.nextInt());
        System.out.println("평균은 " + me.average());
    }
    public void silsoup3(){
        Song s = new Song("Daning Queen", "ABBA", "스웨덴", 1978);
        s.show();
    }
    public void silsoup4(){
        System.out.println("이번달 스케줄 관리 프로그램");
        MonthSchedule ms = new MonthSchedule();
        ms.run();
    }
    public void silsoup5(){
        BallPen bp = new BallPen();
        FountainPen fp = new FountainPen();
        bp.setColor("red");
        bp.setAmount(10);

        fp.setColor("green");
        fp.refill(20);
        System.out.println(fp.getColor());
        System.out.println(fp.getAmount());

    }
    public void silsoup6(){

    }
    public static void main(String[] args){
        EXAM exam = new EXAM();
        exam.silsoup5();

    }
}
