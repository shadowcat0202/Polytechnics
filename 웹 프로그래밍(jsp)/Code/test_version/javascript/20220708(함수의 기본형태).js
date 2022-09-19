function MIDIInput(...items){
    let output = items[0]
    for(const item of items){
        if(output >item){
            output = item
        }
    }
    return output
}