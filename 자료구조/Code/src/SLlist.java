public class SLlist<T> {
    private class Node{
        T data;
        Node next;

        Node(T data){
            this(data, null);
        }
        Node(T data, Node next){
            this.data = data;
            this.next = next;
        }
    }

    private Node head;
    private Node tail;
    private int size;

    public SLlist(){
        this.head = null;
        this.tail = null;
        this.size = 0;
    }

    private Node getNode(int index){
        //인덱스를 가지고 왔으니 판단을 해야한다
        if(index < 0 || index >= this.size) throw new IndexOutOfBoundsException();

        Node temp = head;
        for(int i = 0; i < index; i++){
            temp = temp.next;
        }
        return temp;
    }

    public void addFirst(T value){
        Node newNode = new Node(value);
        newNode.next = this.head;
        this.head = newNode;
        this.size++;

        if(this.head.next == null)   tail = head;
    }
    public void addLast(T value){
        if(this.size == 0){
            addFirst(value);
            return;
        }
        Node newNode = new Node(value);

        tail.next = newNode;
        tail = newNode;
        size++;
    }

    public boolean add(T value){
        addLast(value);
        return true;
    }
    public void add(int index, T value){
        if(index < 0 || index > size)   throw new IndexOutOfBoundsException();
        if(index == 0)  {
            addFirst(value);
            return;
        }
        if(index == size)   {
            addLast(value);
            return;
        }

//        Node prev_Node = getNode(index-1);
//        Node next_Node= prev_Node.next;
//        Node newNode = new Node(value);
//
//        prev_Node.next = null;
//        prev_Node.next = next_Node;
//        newNode.next = next_Node;
        Node prev_Node = getNode(index - 1);
        prev_Node.next = new Node(value, prev_Node.next);

        size++;
    }


    public T remove(){
        if(this.head == null)   throw new IndexOutOfBoundsException();

        T element = head.data;
        Node nextNode = head.next;
        head.data = null;
        head.next = null;
        size--;

        if(size == 0)   tail = null;
        return element;
    }
    public T remove(int index){
        if(index < 0 || index >= size)  throw new IndexOutOfBoundsException();
        if(index == 0)  return remove();

//        Node prev_Node = getNode(index -1);
//        Node remove_Node = prev_Node.next;
//        Node next_Node = remove_Node.next;
        Node prev_Node = getNode(index -1);
        Node remove_Node = prev_Node.next;
        prev_Node.next = prev_Node.next.next;

        T element = remove_Node.data;

//        prev_Node.next = next_Node;

        if(prev_Node.next == null)  tail = prev_Node;

        remove_Node.data = null;
        remove_Node.next = null;
        size--;
        return element;
    }


    public T get(int index){
        return getNode(index).data;
    }

    public boolean contains(T value){return indexOf(value) >= 0;}

    public int indexOf(T value){
        int index = 0;
        for(Node x = head; x != null; x = x.next, index++){
            if(value.equals(x.data))    return index;
        }
        return -1;
    }
    public int size(){
        return this.size;
    }
    public void clear(){
        for(Node x = head; x != null;){
            Node nextNode = x.next;
            x.data = null;
            x.next = null;
            x = nextNode;
        }
        head = tail = null;
        size = 0;
    }
//    public String toString(){
//        StringBuilder sb = new StringBuilder();
//        if(this.size > 0){
//            sb.append("[");
//            for(Node x = head; x != null; x = x.next){
//                sb.append(x.data).append(" ");
//            }
//            sb.delete(sb.length()-1, sb.length());
//            sb.append("]");
//        }else{
//            sb.append("[]");
//        }
//        return sb.toString();
//    }



}
