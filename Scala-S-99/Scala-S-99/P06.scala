object P06{

	def palindrome(list: List[Any]): Boolean = {
		if (list.length <= 1){
			return true
		}
		if (list(0) != list(list.length - 1)){
			return false
		}
		return palindrome(list.slice(1, list.length-1))
	}

	def main(args: Array[String]) {
		val list = List(1, 2, 3, 2, 1)
    	println(palindrome(list))
    	val list2 = List(1, 3, 2)
    	println(palindrome(list2))
	}
}
