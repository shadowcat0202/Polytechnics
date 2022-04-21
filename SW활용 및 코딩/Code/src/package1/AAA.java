package package1;
import package2.CCC;

public class AAA {
    private int AAA_private;
    int AAA_default;
    public int AAA_public;
    protected int AAA_protected;

    public AAA(){
        this.AAA_private = 1;
        this.AAA_default = 2;
        this.AAA_public = 3;
        this.AAA_protected =4;
    }
    void f(){
        BBB b = new BBB();
        CCC c = new CCC();
        StringBuilder sb = new StringBuilder();
        sb.append(b.BBB_protected).append(", ");
        sb.append(b.BBB_public).append(", ");
        sb.append(b.BBB_default);
        //b.BBB_private 접근 불가
        System.out.println(sb);

        sb = new StringBuilder();
        sb.append(c.CCC_public);
        //다름 패키지는 public만 접근 가능
        //다른 형식은 접근 불가
        System.out.println(sb);
    }

}
