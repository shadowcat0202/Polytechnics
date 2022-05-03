public class MATZIP {
    String name;
    String addr;
    int star;

    MATZIP(String name){
        this(name, "", 3);
    }
    MATZIP(String name, String addr){
        this(name, addr, 3);
    }
    MATZIP(String name, String addr, int star){
        this.name = name;
        this.addr =addr;
        this.star = star;
    }
    void showinfo(){
        System.out.println("이름:"+ this.name + "\n주소:" + this.addr + "\n별점:" + this.star);
    }
}
