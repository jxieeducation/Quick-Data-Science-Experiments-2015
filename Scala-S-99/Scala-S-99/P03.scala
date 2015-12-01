object P03{

	def nth(n: Int, list: List[Int]): Int = {
		return list(n)
	}

	def main(args: Array[String]) {
		val list = List(0, 1, 2, 3, 4, 5)
    	println(nth(3, list))
	}
}
