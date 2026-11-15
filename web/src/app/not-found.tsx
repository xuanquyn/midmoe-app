import Link from "next/link";

export default function NotFound() {
  return (
    <div className="mx-auto max-w-md space-y-3 py-16 text-center">
      <h1 className="text-xl font-semibold">Không tìm thấy trang</h1>
      <Link href="/" className="text-primary hover:underline">
        Về trang tổng quan
      </Link>
    </div>
  );
}
