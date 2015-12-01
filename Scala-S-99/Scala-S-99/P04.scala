object P04{

	def getLength(list: List[Int]): Int = {
		return list.length
	}

	def main(args: Array[String]) {
		val list = List(1, 2, 3, 9)
    	println(getLength(list))
	}
}