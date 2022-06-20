import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;
import java.util.Stack;

class Card{
    String shape;
    int number;

    public Card(String _shape, int _number){
        this.shape = _shape;
        this.number = _number;
    }
}
class Player{
    protected ArrayList<Card> cards;  // 추가 확장성을 위한 변수
    protected int sum;    // 강의 자료에서 요구한대로 짜기 위한 최소한의 변수
    protected boolean stay;
    protected boolean dead;
    protected int money;


    Player(){
        this.cards = new ArrayList<>();
        this.sum = 0;
        this.stay = false;
        this.dead = false;
        this.money = 1000;
    }

    void hit(Card card){
        this.cards.add(card);
        this.sum += card.number;
    }
    void stay(){}

    boolean isBurst(){
        return this.sum > 21;
    }
    public void HitOrStay(){
        Scanner sc = new Scanner(System.in);
        System.out.println("continue = 0 or Stay = 1");
        this.stay = sc.nextBoolean();
    }
    public int batting(){
        Scanner sc = new Scanner(System.in);
        System.out.println("배팅할 금액을 입력하세요");
        return sc.nextInt();
    }

}


class Dealer extends Player{
    public Stack<Card> deck;

    Dealer(){
        this.deck = new Stack<>();
        String[] shape = {"Spade", "Heart", "Diamond", "Clover"};
        int[] number = {1,2,3,4,5,6,7,8,9,10,10,10,10};
        for(String s : shape){
            for(int n : number){
                this.deck.push(new Card(s, n));
            }
        }
        Collections.shuffle((List<?>) this.deck);
    }
}

public class BlackJack {
    static void run(){
        int player_number = 0;
        while(true){
            System.out.print("플레이어 수를 입력해라:");
            Scanner sc = new Scanner(System.in);
            player_number = sc.nextInt();
            if (player_number <= 0 || player_number > 5){
                System.out.println("딜러 혼자 못하는데?");
            }else{
                break;
            }
        }

        Player[] players = new Player[player_number];
        Dealer dealer = new Dealer();
        for(Player p : players){
            p.hit(dealer.deck.pop());
            p.hit(dealer.deck.pop());
        }
        dealer.hit(dealer.deck.pop());
        dealer.hit(dealer.deck.pop());
        while(dealer.sum <= 16){
            dealer.hit(dealer.deck.pop());
        }






        boolean playing = true;
        while(playing){
            for(Player p : players){
                if(p.stay)  continue;
                p.hit(dealer.deck.pop());
                if(p.isBurst()){
                    p.dead = true;
                }
            }
        }




    }
    public static void main(String[] args){
        run();
    }
}
