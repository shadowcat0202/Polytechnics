public class Circle {
    int r;
    String name;
    public Circle(int r){
        this.r=r;
        this.name = "none";
    }
    public Circle(int r, String name){
        this(r);
        this.name = name;
    }

    public void set(int r){
        this.r = r;
    }

    public double getArea(){
        return 3.14 * r * r;
    }


    public void info(){
        System.out.println("r:"+ this.r + "\tname:"+this.name);
    }


}
