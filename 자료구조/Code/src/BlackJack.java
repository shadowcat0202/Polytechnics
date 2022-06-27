import java.util.*;
import java.util.Stack;

class Card{
    enum SHAPE {Spade, Heart, Diamond, Club};
    SHAPE shape ;
    int number;

    public Card(SHAPE _shape, int _number){
        this.shape = _shape;
        this.number = _number;
    }
    void show(){
        String alphaNumber = "";
        if(number == 1) alphaNumber = "A";
        else if(number == 11) alphaNumber = "J";
        else if(number == 12) alphaNumber = "Q";
        else if(number == 13) alphaNumber = "K";
        else alphaNumber = Integer.toString(number);

        System.out.print("[" + shape + ", " + alphaNumber + "]");
    }


}
class Player{
    private String name;
    private int credit;
    private int betMoney;
    private LinkedList<Card> cardsInHand;
    Player(String name, int credit){
        this.name = name;
        this.credit = credit;
        cardsInHand = new LinkedList<>();
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getCredit() {
        return credit;
    }

    public void setCredit(int credit) {
        this.credit = credit;
    }

    public int getBetMoney() {
        return betMoney;
    }
    public LinkedList<Card> getCardsInHand(){
        return cardsInHand;
    }
    public void setBetMoney(int betMoney) {
        this.betMoney = betMoney;
    }


    void clearHand(){
        cardsInHand.clear();
    }
    void getCard(Card c){
        cardsInHand.add(c);
    }
    void showPlayInfo(){
        System.out.println("=====================");
        System.out.println("Player:" + name + "\n보유금액:" + credit);
        for(Card c : cardsInHand){
            c.show();
        }
        System.out.println("\n=======================");
    }

    void bet(int money){
        if(money > credit){
            System.out.println("보유 금액 부족");
            System.out.println("All in으로 간주합니다.");
            betMoney = credit;
        }
        else{
            betMoney = money;
        }
    }
    int getScore(){
        int score = 0;
        int num_of_ace = 0;

        for(Card c : cardsInHand){
            if(c.number == 1) ++num_of_ace;
//            else if(c.number > 10)  score += 10;
//            else score += c.number;
            else score += Math.min(c.number, 10);
        }

        if(num_of_ace > 0){
            int max = 11 + num_of_ace - 1;

            if((score + max) < 22) score += max;
            else score += num_of_ace;
        }
        return score;
    }

    public void win() {
        credit += getBetMoney();
    }

    public void lose() {
        credit -= getBetMoney();
    }
}
class Dealer extends Player{
    private LinkedList<Card> deck;

    Dealer(String name, int credit){
        super(name, credit);
        deck = new LinkedList<>();
    }
    void makeDeck(){
        deck.clear();
        for(Card.SHAPE s: Card.SHAPE.values()){
            for(int i=1; i <= 13; i++){
                Card c = new Card(s, i);
                deck.add(c);
            }
        }
        Collections.shuffle(deck);
    }
    Card deal(){
        return deck.pop();
    }
    void showPlayInfo(){
        System.out.println("=====================");
        System.out.println(getName() + "\n보유금액:" + getCredit());
        for(Card c : getCardsInHand()){
            c.show();
        }
        System.out.println("\n=======================");
    }

    void dealerRule(){
        int score = getScore();
        if(score < 17){
            getCard(deal());
        }
    }


}

public class BlackJack {
    enum GameStatus {INIT, MAKEUP, PLAY, CAL_SCORE, CONTINUE, END};
    private GameStatus status;
    private Dealer dealer;
    private ArrayList<Player> players;
    private final int MAX_PLAYER = 4;
    private
    BlackJack(){
        status = GameStatus.INIT;
        dealer = new Dealer("Dealer", 1000);
        players = new ArrayList<>();
    }
    boolean nameExist(String name){
        for(Player p : players){
            if(p.getName().equals(name))    return true;
        }
        return false;
    }
    void init(){
        System.out.println("init game");
        dealer.makeDeck();
        dealer.clearHand();
        for (Player p : players){
            p.clearHand();
        }
        status = GameStatus.MAKEUP;
    }
    void makeUp(){
        System.out.println("makeUp");
        Scanner sc = new Scanner(System.in);

        while(players.size() < MAX_PLAYER){
            int slot = MAX_PLAYER - players.size();
            System.out.println("남은 자리:" + slot);
            System.out.println("참가자 추가? (y/n):");
            String ans = sc.next();

            if(ans.equals("N") || ans.equals("n"))  break;

            System.out.print("사용자 이름:");
            String name = sc.next();
            if(nameExist(name)) {
                System.out.println("플레이어 이름이 존재합니다 다른 이름을 사용해 주세요.");
                continue;
            }
            System.out.print("보유 금액:");
            int credit = sc.nextInt();

            players.add(new Player(name, credit));
            System.out.println("플레이어 [" + name + "]님이 참가 하였습니다.");
        }
        if(players.size() > 0) {
            status = GameStatus.PLAY;
        }
        else{
            System.out.println("참가자가 없어서 종료합니다.");
            status = GameStatus.END;
        }
    }

    void play(){
        System.out.println("play");

        Scanner sc = new Scanner(System.in);
        for(Player p : players){
            System.out.print("플레이어 " + p.getName() + " 베팅 금액을 입력하세요:");
            int bet = sc.nextInt();
            p.bet(bet);
        }

        status = GameStatus.CAL_SCORE;
        for(Player p : players){
            p.getCard(dealer.deal());
            p.getCard(dealer.deal());
        }
        dealer.getCard(dealer.deal());
        dealer.getCard(dealer.deal());

        dealer.showPlayInfo();

        for(Player p : players){
            p.showPlayInfo();
            String ans = "";
            do{
                System.out.println("플레이어 " + p.getName() + " Hit or Stay?(h/s):");
                ans = sc.next();
                if(ans.equals("s") || ans.equals("S"))  break;

                p.getCard(dealer.deal());
                p.showPlayInfo();
                int score = p.getScore();
                if(score > 21){
                    System.out.println("21 초과");
                    break;
                }
            }while(ans.equals("H") || ans.equals("h"));
        }
        dealer.dealerRule();
    }

    void calScore(){
        int dealerScore = dealer.getScore();
        System.out.println("딜러의 점수는 " + dealerScore);
        if(dealerScore > 21){
            System.out.println("모든 플레이어가 승리");
            for(Player p : players) p.win();
        }
        else{
            for(Player p : players){
                int playScore = p.getScore();
                if (playScore > 21) {
                    System.out.println("플레이어 " + p.getName() + " 버스트입니다.");
                    p.lose();
                }
                else {
                    if (playScore > dealerScore) {
                        System.out.println("플레이어 " + p.getName() + " win");
                        p.win();
                    }
                    else if (dealerScore > playScore) {
                        System.out.println("플레이어 " + p.getName() + " lose");
                        p.lose();
                    }
                    else {
                        System.out.println("플레이어 " + p.getName() + " 무승부");
                    }
                }

            }
        }

        status = GameStatus.CONTINUE;
    }

    void checkContinue(){
        Iterator<Player> iter = players.iterator();
        Scanner sc = new Scanner(System.in);
        while(iter.hasNext()){
            Player next = iter.next();
            if(next.getCredit() <= 0){
                System.out.println("[" + next.getName() + "]님은 크래딧이 부족하여 플레이 할 수 없습니다.");
                iter.remove();
            }
            else{
                System.out.println("["+next.getName()+"]님의 남은 크래딧 " + next.getCredit() + " 입니다.");
                System.out.println("계속 하시겠습니까?(y/n)");
                String ans = sc.next();
                if(ans.equals("n") || ans.equals("N")){
                    System.out.println("["+next.getName() + "]님은 나가셨습니다.");
                    iter.remove();
                }
            }
        }

        if(players.size() > 0){
            status = GameStatus.INIT;
            System.out.println("플레이어가 남아있어 게임을 계속합니다.");
        }
        else{
            System.out.print("남은 플레이어가 없습니다 추가 플레이어를 받으시겠습니까? (y/n)");
            String ans = sc.next();
            if(ans.equals("n") || ans.equals("N"))
                status = GameStatus.END;
            else
                status = GameStatus.INIT;
        }

    }
    void startGame(){
        System.out.println("Game Start");
        while(status != GameStatus.END){
            switch (status){
                case INIT:
                    init();
                    break;
                case MAKEUP:
                    makeUp();
                    break;
                case PLAY:
                    play();
                    break;
                case CAL_SCORE:
                    calScore();
                    break;
                case CONTINUE:
                    checkContinue();
                    break;

            }
        }

    }
    public static void main(String[] args){
        BlackJack bj = new BlackJack();
        bj.startGame();
    }
}
